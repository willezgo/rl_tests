# Random RL experiments

This repo is a set of my personal random experimentations about RL without any real purpose other than learning and tinkering.

It's a bit of a mess sometimes, so reader caution advised.


## Summary of all the files:

- Dockerfile : good starting point to setup a Docker image with all the dependencies (notably Gym and stable-baselines3)
- dollmod.py : this implements the "Red Light Green Light" game as an environment module.
- entrypoint.sh : entrypoint to the Docker container, I frequently changed which script it runs depending on the experiments I wanted to run. But if a video is being recorded, it's important to call "xvfb-run" to start a virtual X server.
- envmod.py : base classes to build environment modules (plugins).
- lib.py : a few utilities (loading, wrappers, video recording...)
- mybipedal.py : this is copy/pasted from the OpenAI Gym Bipedal environment, but modified to include a "module" system that allows to add new rules to this game.
- opti.py : experiment to search and evaluate different hyperparameters.
- run.py : a simple agent training / evaluation loop
- test_record.py : another training / evaluation loop that periodically records videos of the agent progress.
- test.py : another training loop, this time for Atari environments.
