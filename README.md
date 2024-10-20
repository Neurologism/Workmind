# Workmind

Welcome to the training repository for our innovative educational AI project, [Whitemind](https://github.com/Neurologism/whitemind)! Our goal is to empower users to build artificial intelligence systems using a visual, block-based interface similar to Scratch or LEGO, making AI development accessible and fun for everyone.

## Table of Contents

Already know where you're going? We've got you covered.

- [Technical Stack](#technical-stack)
- [Installation](#installation)
- [Configuration](#configuration)

## Technical Stack

Workmind is built using a modern and robust technology stack, ensuring scalability, performance, and ease of development:

### Python 3.12

Python 3.12 continues to solidify Python's reputation as a leading programming language in the field of deep learning and data science. With its user-friendly syntax and extensive libraries, Python enables researchers and developers to implement complex algorithms efficiently. Enhancements in performance, typing, and error messages in this version further streamline the coding experience, making it an ideal choice for developing machine learning models and deep learning applications.

### Numpy

NumPy is a foundational library for numerical computing in Python, providing support for large, multi-dimensional arrays and matrices. It serves as the backbone for many other libraries, including those used in deep learning, by offering efficient operations for array manipulation, mathematical functions, and linear algebra. NumPy's optimized performance ensures that complex computations can be performed rapidly, which is essential for training deep learning models on large datasets.

### Tensorflow

TensorFlow is an open-source deep learning framework developed by Google that has become a standard in the industry for building and deploying machine learning models. It offers a robust platform for developing neural networks and supports both CPU and GPU computation. TensorFlow's flexibility allows developers to create complex architectures with ease, making it suitable for tasks ranging from simple regression to advanced deep learning applications like image recognition and natural language processing.

### Keras

Keras is a high-level neural networks API that runs on top of TensorFlow, simplifying the process of building and training deep learning models. Designed for ease of use and modularity, Keras allows developers to quickly prototype and iterate on their models with minimal code. Its user-friendly interface makes deep learning more accessible, enabling beginners to experiment with neural networks while still offering advanced features for experienced practitioners. Keras enhances TensorFlow’s capabilities by providing a streamlined approach to model creation and evaluation.

## Installation

### 1. Install Required Tools

You can either install Python using your package manager [or downloading it](https://www.python.org/downloads/) if you haven't installed it already. Afterwards, verify that you've installed the correct version of python. 

```bash
python --version # should print something like `Python 3.12.6`
pip --version # should print something like `pip 24.2 from /path/to/pip (python 3.12)`
```

### 2. Installing MongoDB

If you have access to an already running instance of mongodb, you can skip this step. Otherwise:

To proceed, you need to install MongoDB Community Edition on your local machine. MongoDB offers official installation guides based on your operating system. Follow the instructions for your platform to ensure a correct setup. Select the appropriate tutorial from the list below:

| Platform | Link                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linux    | [Red Hat or CentOS](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-red-hat/) <br> [Ubuntu](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/) <br> [Debian](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-debian/) <br> [SUSE](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-suse/) <br> [Amazon Linux](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-amazon/) |
| maxOS    | [macOS]()                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Windows  | [Windows]()                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Docker   | [Docker]()                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

### 3. Cloning the Workmind Repository

Navigate to or create a directory where you want to store Workmind and run the following commands. Remember to insert your MongoDB login data into the `MONGO_URI` environment variable.

```bash
git clone https://github.com/Neurologism/Workmind.git
cd Workmind
npm install
echo "MONGO_URI='mongodb://user:password@localhost:27017'" >> .env
```

You also need to create a virtual environment locally. Make sure you've installed the virtualenv python package and run:

```bash
virtualenv venv
source venv/bin/activate
```

To install all packages needed, run the following commands. Some packages are pretty big, so this step could take a while. Consider to get a coffee. 

```bash
pip install -r requirements-dev.txt # or requirements.txt in production
```

## Configuration

### Basics

Configuration data is stored as environment variables in the [`.env`](/.env) file.
The file is part of the [`.gitignore`](/.gitignore), thus you will have to create it manually if you want to specify environment variable values different from the default values.
Below, you can find an example configuration utilizing some environment variables.

### Examples

Example configuration:

```bash
# /.env
MONGO_URI='mongodb://user:password@hostname:port'
LOG_LEVEL="debug"
```
