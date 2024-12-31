# Analytics Dashboard for Service Providers

This repository contains essential code to setup an analytics dashboard for a website that connects service seekers and service providers called <i>AppointUs.</i> Follow the instructions below to successfully setup the codebase and run the app.

<b>Step 1: Clone this repository</b><br/>
Clone this repository in your local system using git.<br/><br/>
<b>Step 2: Setup a virtual environment</b><br/>
Navigate to the local git repository and setup a virtual environment. Make sure to activate the venv.<br/><br/>
<b>Step 3: Install the necessary packages</b><br/>
Run the following command:<br/>
```pip install -r requirements.txt```<br/><br/>
<b>Step 4: Execute the following files</b><br/>
The following files will initialize the database and add sample data into the tables.<br/>
```python db_init.py```<br/>
```python db_add.py```<br/>
```python db_add2.py```<br/><br/>

<b>Step 5: Run the streamlit app</b><br/>
```streamlit run app.py```<br/>
You can view the visual interactive dashboard through the link provided by streamlit<br/>

