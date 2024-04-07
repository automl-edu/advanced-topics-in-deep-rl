# Advances Topics in Deep RL

This repository contains the lecture materials. There are two main directories:
- adrl contains constructors for our main three settings and small examples
- lecture contains the lecture PDFs and you will add your seminar contributions via PR there as well


You should install the repository as below to run experiments in our settings. You should be able to interact with the continual environment as with any other env. For the multi-agent interface see [Petting Zoo](https://pettingzoo.farama.org/) and for offline [Minari](https://minari.farama.org/main/).

## Installation

Ideally, you'll follow these instructions to create a fresh conda environment and then install for usage. That should allow you to run the examples and use the constructor functions for all three settings.
The dev option simply enables formatting in case you're interested in using that.

```
git clone https://github.com/automl/adrl.git
cd adrl
conda create -n adrl python=3.10
conda activate adrl

# Install for usage
make install

# Install for development
make install-dev
```
