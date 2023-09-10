This project contains an application for using EAST for text detection.

- text_detection_app.py contains the main application
- example_images contains some example images for using with the app.
Other can in any other location on your computer can of course be used as well.
The start.jpg in the directory shall not be removed.
- temp_image is a directory in which the current imaged to be worked at is loaded with the gui. 
This folder shall not be modified or removed.
- EAST.pb contains the pretrained model for text detection

### Setup

In order to setup the application it is recommended to use a virtual environment. With the following steps this can be done.

1. **Create a Virtual Environment:**

   First, create a new virtual environment using the `venv` module. Here named `venv`.
   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment:**

   On most systems, you can activate the virtual environment with the following commands.
   Cd into the project directory and execute:

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source venv/bin/activate
     ```

3. **Install Packages from the Requirements File:**

   Install the packages from the requirements file with:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app using this environment**