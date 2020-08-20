import functools
import numpy as np

from absl import app, flags
from trax import data, models, shapes, fastmath
from trax import layers as tl, optimizers as opt
from trax.supervised import lr_schedules as lr, Trainer
from trax.rl import distributions, rl_layers
from trax.rl.training import RLTrainer, remaining_evals
from trax.rl.task import RLTask

FLAGS = flags.FLAGS
flags.DEFINE_integer('warmup_steps', 100, 'Warmup steps.')


# Copyright 2020 The Trax Authors.
# see https://github.com/google/trax/pull/908/files 
class ValueTrainer(RLTrainer):
  """Trainer that uses a deep learning model for policy.
  Q-value based training.
  """

  def __init__(self, task, value_body=None, value_optimizer=None,
               value_lr_schedule=lr.multifactor, value_batch_size=64,
               value_train_steps_per_epoch=500, value_evals_per_epoch=1,
               value_eval_steps=1, n_eval_episodes=0,
               only_eval=False, max_slice_length=1, output_dir=None, **kwargs):
    """Configures the value trainer.
    Args:
      task: RLTask instance, which defines the environment to train on.
      value_body: Trax layer, representing the body of the value model.
          functions and eval functions (a.k.a. metrics) are considered to be
          outside the core model, taking core model output and data labels as
          their two inputs.
      value_optimizer: the optimizer to use to train the policy model.
      value_lr_schedule: learning rate schedule to use to train the policy.
      value_batch_size: batch size used to train the policy model.
      value_train_steps_per_epoch: how long to train policy in each RL epoch.
      value_evals_per_epoch: number of policy trainer evaluations per RL epoch
          - only affects metric reporting.
      value_eval_steps: number of policy trainer steps per evaluation - only
          affects metric reporting.
      n_eval_episodes: number of episodes to play with policy at
        temperature 0 in each epoch -- used for evaluation only
      only_eval: If set to True, then trajectories are collected only for
        for evaluation purposes, but they are not recorded.
      max_slice_length: the maximum length of trajectory slices to use.
      output_dir: Path telling where to save outputs (evals and checkpoints).
      **kwargs: arguments for the superclass RLTrainer.
    """
    super(ValueTrainer, self).__init__(
        task,
        n_eval_episodes=n_eval_episodes,
        output_dir=output_dir,
        **kwargs
    )
    self._value_batch_size = value_batch_size
    self._value_train_steps_per_epoch = value_train_steps_per_epoch
    self._value_evals_per_epoch = value_evals_per_epoch
    self._value_eval_steps = value_eval_steps
    self._only_eval = only_eval
    self._max_slice_length = max_slice_length
    self._policy_dist = distributions.create_distribution(task.action_space)

    # Inputs to the policy model are produced by self._policy_batches_stream.
    self._inputs = data.inputs.Inputs(
        train_stream=lambda _: self.value_batches_stream())

    # policy_model = functools.partial(models.Policy, body=value_body)

    # policy_model = functools.partial(
    #    policy_model,
    #    policy_distribution=self._policy_dist,
    # )

    # We will be optimizing only according to the value
    # but we have to copy weights from the trainer to
    # the to eval/collect policy, hence we need proper
    # output shapes.
    joint_model = functools.partial(models.PolicyAndValue, body=value_body)
    joint_model = functools.partial(
        joint_model,
        policy_distribution=self._policy_dist,
    )

    # value_model = functools.partial(models.Value, body=value_body)

    # This is the value Trainer that will be used to train the value model.
    # * inputs to the trainer come from self.value_batches_stream
    # * outputs, targets and weights are passed to self.value_loss
    self._value_trainer = Trainer(
        model=joint_model,
        optimizer=value_optimizer,
        lr_schedule=value_lr_schedule(),
        # We would be glad to use tl.L2Loss, but
        # this make an assumption that we have just one head
        # loss_fn=tl.L2Loss(),
        loss_fn=self.value_loss,
        inputs=self._inputs,
        output_dir=output_dir,
        # metrics={'value_loss': tl.L2Loss()}
        metrics={'value_loss': self.value_loss,}
    )
    self._collect_model = tl.Accelerate(
        joint_model(mode='collect'), n_devices=1)
    policy_batch = next(self.value_batches_stream())
    self._collect_model.init(shapes.signature(policy_batch))
    # Not collecting stats
    self._eval_model = tl.Accelerate(
        joint_model(mode='collect'), n_devices=1)
    self._eval_model.init(shapes.signature(policy_batch))
    if self._task._initial_trajectories == 0:
      self._task.remove_epoch(0)
      self._collect_trajectories()

  @property
  def value_loss(self):
    """Value loss."""
    return NotImplementedError

  # @property
  # def value_metrics(self):
  #   return {'value_loss': self.value_loss}

  def value_batches_stream(self):
    """Use self.task to create inputs to the policy model."""
    return NotImplementedError

  def policy(self, trajectory, temperature=1.0):
    """Chooses an action to play after a trajectory."""
    policy_model = self._collect_model
    if temperature != 1.0:  # When evaluating (t != 1.0), don't collect stats
      policy_model = self._eval_model
      policy_model.state = self._collect_model.state

    policy_model.weights = self._value_trainer.model_weights
    tr_slice = trajectory[-self._max_slice_length:]
    trajectory_np = tr_slice.to_np(timestep_to_np=self.task.timestep_to_np)
    # Add batch dimension to trajectory_np and run the model.
    # Since it is a joint model, it will produce a tuple of logits and value.
    pred = policy_model(trajectory_np.observations[None, ...])[0]
    # Pick element 0 from the batch (the only one), last (current) timestep.
    pred = pred[0, -1, :]
    sample = self._policy_dist.sample(pred, temperature=temperature)
    result = (sample, pred)
    if fastmath.backend_name() == 'jax':
      result = fastmath.nested_map(lambda x: x.copy(), result)
    return result

  def train_epoch(self):
    """Trains RL for one epoch."""
    # When restoring, calculate how many evals are remaining.
    n_evals = remaining_evals(
        self._value_trainer.step,
        self._epoch,
        self._value_train_steps_per_epoch,
        self._value_evals_per_epoch)
    for _ in range(n_evals):
      self._value_trainer.train_epoch(
          self._value_train_steps_per_epoch // self._value_evals_per_epoch,
          self._value_eval_steps)

  def close(self):
    self._value_trainer.close()
    super().close()


class DQNTrainer(ValueTrainer):
  """Trains a value model using DQN on the given RLTask."""

  @property
  def value_loss(self):
    """Value loss - so far generic for all A2C."""
    def f(dist_inputs, values, returns):
      del dist_inputs
      return rl_layers.ValueLoss(values, returns, 1)
    return tl.Fn('ValueLoss', f)

  def value_batches_stream(self):
    """Use self.task to create inputs to the policy model."""
    for np_trajectory in self._task.trajectory_batch_stream(
        self._value_batch_size,
        epochs=[-1],
        max_slice_length=self._max_slice_length,
        sample_trajectories_uniformly=True):
      ret = np_trajectory.returns
      ret = (ret - np.mean(ret)) / np.std(ret)  # Normalize returns.
      # We return a pair (observations, normalized returns) which is
      yield (np_trajectory.observations, ret)


def main(_):
    """Test-runs joint PPO on CartPole."""

    task = RLTask('CartPole-v0', initial_trajectories=0, max_steps=2)
    value_body = lambda mode: tl.Serial(tl.Dense(64), tl.Relu())

    lr_schedule = lambda: lr.multifactor(constant=1e-2, 
      warmup_steps=FLAGS.warmup_steps, 
      factors='constant * linear_warmup')

    trainer = DQNTrainer(
        task,
        value_body=value_body,
        value_optimizer=opt.Adam,
        value_lr_schedule=lr_schedule,
        value_batch_size=4,
        value_train_steps_per_epoch=2,
        n_trajectories_per_epoch=5,)
    trainer.run(2)

    print("Average returns: %s" % trainer.avg_returns)


if __name__ == '__main__':
  app.run(main)
