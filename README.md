# Prospero

Prospero is a collection of tutorials for applying machine learning and optimization to enterprise operations.  The intent is enable researchers to run a production-ready reference model on [Google Cloud AI](https://cloud.google.com/ai-platform) with [MLFlow](https://mlflow.org/) tracking in under five minutes.

In development:
* [Reinforcement Learning for Pricing](https://github.com/jordanrule/prospero/tree/master/pricing)

All credit to the authors of:
* [Caliban](https://github.com/google/caliban) for Google Cloud AI production tooling
* [Trax](https://github.com/google/trax) for reference JAX implementations
* [TensorHouse](https://github.com/ikatsov/tensor-house) for reference enterprise model implementations

## Quickstart

[Install Docker](https://hub.docker.com/editions/community/docker-ce-desktop-mac), make sure it's running, then install Caliban (you'll need [Python >= 3.6](https://www.python.org/downloads/mac-osx)):

```bash
pip install caliban
```

Train a pricing model on your local machine:

```bash
git clone https://github.com/jordanrule/prospero.git && cd pricing/
caliban run --experiment_config experiment.json --xgroup pricing_tutorial --nogpu pricing.py
```

Train a pricing model on [Google Cloud](https://caliban.readthedocs.io/en/latest/getting_started/cloud.html):

```bash
caliban cloud run --experiment_config experiment.json --xgroup pricing_tutorial --nogpu pricing.py
```

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
