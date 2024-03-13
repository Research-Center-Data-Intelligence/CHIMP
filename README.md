# CHIMP

CHIMP (Continuous Hypothesis and Information Mending Pipeline) is a project that aims to give a reference for operating
a continual AI pipeline. It was originally part of a bachelor thesis project.

## Components

- **serving:** Responsible for serving the model for inference. This is done using a REST API.
- **experimentation:** Responsible for training the model and logging the results. The resulting model and logs are
  stored into MLFlow.
- **ml-frontend:** A web-application that is responsible for managing the interaction with the user (note: while it is
  called "front-end", it also contains a back-end component in the form of a Flask API that communicates with the rest
  of the CHIMP system).
- **mlflow-tracking:** MLFlow is used to track the different models and log metrics for these models.

The table below shows which ports are used by the different components. The "Local Dev Port" is the port that is used when a component is ran on the host instead of in Docker.

| Component       | Local Dev Port | Docker Host Porst | Docker Internal Ports |
|-----------------|----------------|-------------------|-----------------------|
| serving         | 5254           | 5254              | 8000                  |
| experimentation | 5253           | 5253              | 8000                  |
| ml-frontend     | 5252           | 5252              | 8000                  |
| mlflow-tracking | n.a.           | 8999              | 8999                  |

```mermaid
graph RL
;
    subgraph Application
        afe[Emotion recognition front-end<br/>- HTML/CSS/JS] --> abe[Emotion recognition back-end<br/>- Python/Flask];
    end
    abe --> exp[Experimentation service<br/>- Python/Flask<br/>- TalosML/Tensorflow];
    abe --> srv[Serving service<br/>- Python/Flask];
    subgraph Services
        exp --> mlf[Tracking<br/>- MLFlow];
        srv --> mlf;
        mlf --> db[Database<br/>- SQLite];
        mlf --> fs[File storage<br/>- Filesystem];
    end
```

## Development setup

1. Fork this repository and clone the fork to your local machine.
2. Install Docker and Docker Compose.
3. Run `docker-compose up` in the root of the repository.
4. Open your browser and navigate to `http://localhost:5252`.
5. Allow the CHIMP front-end to use your webcam.
6. Click on browse and select the calibration-data.zip (which can be downloaded on Teams).
7. Click on "Train New Model" and wait for the training to finish.
    - You can monitor the training progress in the terminal by running `docker logs -f experimentation-server`.
8. Once the training is finished, navigate to `http://localhost:8999` (MLFlow).
9. Click on "Models" and select "onnx emotion model".
10. Click on the last model and set the model to "Production" by changing the "stage" setting at the top of the screen.
11. Click on the second to last model and sit the model to "Staging" by changing the "stage" setting at the top of the screen.
12. Refresh the CHIMP front-end. You should now see percentages of the detected emotions.

To run the GPU enabled version of CHIMP use the "gpu" profile as follows: `docker-compose --profile gpu up -d`. To only
run MLFlow (e.g. when running the Python code directly on the host), use the "mlflow" profile as
follows: `docker-compose --profile mlflow up -d`. Please note that the `--profile <PROFILE_NAME>` should come before any
further commands, such as `up -d`. On some setups, to use the default profile (denoted by an empty string, or ''), it should be explicitly included in the call, for example: `docker-compose --profile '' up -d`.

### Local development setup (on host outside of Docker)
To run the Python/Flask based CHIMP components outside of Docker (for example, when you want to run a component with a debugger attached), you can use the following steps:
- Run the MLFlow service in Docker using `docker-compose --profile mlflow up -d`
- Create a virtual environment for each component with the required Python versions 
  - Experimentation (Python 3.9) `python3.9 -m venv experimentation/env`
  - Serving (Python 3.9) `python3.9 -m venv serving/env`
  - ML-frontend (Python 3.11) `python3.11 -m venv ml-frontend/env`
- Activate the virtual environment for each component
  - Experimentation `source experimentation/env/bin/activate`
  - Serving `source serving/env/bin/activate`
  - ML-frontend `source ml-frontend/env/bin/activate`
- Install the dependencies for each component using said components' `requirements.txt` file while the right virtual environment is activated:
  - Experimentation `pip install -r experimentation/requirements.txt`
  - Serving `pip install -r serving/requirements.txt`
  - ML-frontend `pip install -r ml-frontend/requirements.txt`
- Each component can now be ran as usual
  - In PyCharm, a "Compound" run configuration can be used to run all the configurations for each components at once (including debugging)
