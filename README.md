# Prospero

Prospero is a collection of tutorials for applying machine learning and optimization to enterprise operations.  The intent is enable researchers to run a production-ready reference model on [Google Cloud AI](https://cloud.google.com/ai-platform) with [MLFlow](https://mlflow.org/) tracking in under five minutes.

In development:
* [Reinforcement Learning for Pricing](https://github.com/jordanrule/prospero/tree/main/pricing)

All credit to the authors of:
* [Caliban](https://github.com/google/caliban) for Google Cloud AI production tooling
* [Trax](https://github.com/google/trax) for reference JAX implementations
* [TensorHouse](https://github.com/ikatsov/tensor-house) for reference enterprise model implementations

## Quickstart

[Install Docker](https://hub.docker.com/editions/community/docker-ce-desktop-mac), make sure it's running, then install Caliban (you'll need [Python >= 3.6](https://www.python.org/downloads/mac-osx)):

```bash
pip install caliban
```

To run Caliban, you will need the [Google Cloud SDK](https://cloud.google.com/sdk/) installed to login:

```bash
gcloud auth login
```

Train a pricing model on your local machine:

```bash
git clone https://github.com/jordanrule/prospero.git && cd pricing/
caliban run --experiment_config experiment.json  --nogpu pricing.py
```

Train a pricing model on a Google Cloud GPU:

```bash
caliban cloud --experiment_config experiment.json pricing.py
```

## Contributing

The vision of Prospero is that models contained should:
* Easily deploy on Google Cloud for immediate integration and iteration
* Purify stateful code (see [here](https://sjmielke.com/jax-purify.htm) or [here](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html))
* Solve a business problem

If that sounds like something you would be interested in helping guide, reach out to Jordan in the #ai-ml channel of [Docker Slack](https://www.docker.com/docker-community).

### Dramatic Epilogue

<p>
<img style="float: right;" align="right" src="https://upload.wikimedia.org/wikipedia/commons/6/6a/William_Hamilton_Prospero_and_Ariel.jpg" width="280">

> “Our revels now are ended. These our actors, \
> As I foretold you, were all spirits, and \
> Are melted into air, into thin air: \
> And like the baseless fabric of this vision, \
> The cloud-capp'd tow'rs, the gorgeous palaces, \
> The solemn temples, the great globe itself, \
> Yea, all which it inherit, shall dissolve, \
> And, like this insubstantial pageant faded, \
> Leave not a rack behind. We are such stuff \
> As dreams are made on; and our little life \
> Is rounded with a sleep.”
>
> -- <cite>Shakespeare, The Tempest</cite>
</p>
