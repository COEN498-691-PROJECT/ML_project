# Development of a Machine Learning Pipeline for Human Activity Recognition
 ## Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [TensorFlow TFX Pipeline](#tensorflow-tfx-pipeline)
  - [Gradio Deployment](#gradio-deployment)
  - [Collaborators](#collaborators)

## Getting Started

To get this project up and running on your local machine, please follow these steps.  
Note: An older python version may be required to run tfx pipeline, see TensorFlow TFX Pipeline.

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/COEN498-691-PROJECT/ML_project.git

2.  **Navigate to the project directory**
    ```bash
    cd ML_project
    ```

3.  **Install dependencies**
    ```bash
    pip install
    ```


### TensorFlow TFX Pipeline
Video demo of the tfx pipeline: https://drive.google.com/file/d/1U2wrudpB5Su2tluR3CdlRmcPin3_H0_U/view?usp=sharing

For the best display, the file should be run on as instructed below. The program was originally opened on VS Code, then Jupyter Notebook. Google Collab may not run the program properly, since it defaults to a newer python version.

1.  **Check python version**  
Ensure it runs a version of python 3.9 (newer models may not work with tfx, we use Python 3.9)
    ```bash
    python --version
    ```

2.  **Run virtual environment, ex. .venv**
    ```bash
    source .venc/bin/activate
    ```

3.  **Run the first cell with the imports**

4.  **Install dependencies**
    ```bash
    pip install
    ```

5.  **Run jupyter notebook**  
This opens jupyter notebook locally on browser.
    ```bash
    jupyter notebook
    ```

6.  **Go to the tfx folder**

7.  **Open "TFX_pipeline.ipynb" file**

8.  **Select Trust notebook**

9.  **Run all cells or each cell one by one**  
It will take a few minutes to run all the cells. Warnings may appear which can be ignored for the current project. Each tfx component has an interactive table created with interactive context. The trainer has an interactive TensorBoard.


### Gradio Deployment
Video demo of the gradio: https://drive.google.com/file/d/1cHgq4xRLhLxZSvIS9ztSxmdrOxEt4NLL/view?usp=sharing

Quick Gradio Start Guide:
Please follow the steps below to set up and launch the application in your environment.

1.  **Ensure Environment Setup**  
Run the first code block to install the necessary library (gradio)

2.  **Mount Google Drive**  
If the environment setup step did not automatically prompt or complete the process, run the designated code block, if necessary, to ensure Google Drive is mounted. This is required for accessing the application's model and dataset files.

3.  **Launch the Gradio Interface**  
Run the final code block in your notebook. This code block defines the structure of the Gradio interface and starts the application service.

4.  **Open the Application in the Browser**  
Once the Gradio launch command successfully executes in the notebook, Gradio will print a public URL in the console output.  

Example Output:  

Running on public URL: https://[unique_id].gradio.live
Click this link (https://[unique_id].gradio.live) to access and use the application in your web browser. 




### Collaborators

| Name   | Username |
| -------- | ------- |
| Laurie Anne Laberge  | P4SS3-P4RT0UT   |
| Laura Hang | ssugarcane    |
| Yunxing Tao   | aurora-tao   |
| Mahsa Khatibi   | thisisminecraftvillager   |


---





