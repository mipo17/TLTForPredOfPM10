# TLTForPredOfPM10

# Setting up a Python Virtual Environment and Installing Dependencies

This guide explains how to create a virtual environment for Python 3.10.1 and install the required dependencies from a text file.

## Prerequisites

Make sure you have the following installed on your system:

- Python 3.10.1
- pip (comes pre-installed with Python)

To check if Python is installed and confirm the version, run:

```bash
python --version
```

Ensure the output shows Python 3.10.1. If not, download and install Python 3.10.1 from the [official Python website](https://www.python.org/).

## Steps to Set Up the Environment

1. **Create a Virtual Environment**

   Use the following command to create a virtual environment:

   ```bash
   python -m venv venv
   ```

   This will create a directory named `venv` in your current folder. It contains the Python executable and libraries.

2. **Activate the Virtual Environment**

   - On **Windows**:

     ```bash
     venv\Scripts\activate
     ```

   - On **macOS/Linux**:

     ```bash
     source venv/bin/activate
     ```

   After activation, you should see the virtual environment's name (e.g., `(venv)`) at the beginning of your terminal prompt.

3. **Install the Required Packages**

   Ensure you have a requirements file (e.g., `requirements.txt`) containing the list of dependencies. To install the packages, run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**

   To confirm the packages were installed successfully, run:

   ```bash
   pip list
   ```

   This will display all installed packages in the virtual environment.

## Deactivating the Virtual Environment

When you are done working in the virtual environment, deactivate it by running:

```bash
deactivate
```

This will return you to your system's global Python environment.

## Notes

- Always activate the virtual environment before running or developing Python applications to ensure you are using the correct dependencies.
- To recreate the environment on another machine, you can share the `requirements.txt` file and follow the steps above.

---

For additional help or troubleshooting, refer to the [Python Virtual Environments Documentation](https://docs.python.org/3/library/venv.html).
