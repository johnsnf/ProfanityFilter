For this specific problem, requires python3.13

To create a python virtual environment:
python3 -m venv {NAME OF VIRTUAL ENVIRONMENT}


To activate environment:
source {NAME OF VIRTUAL ENVIRONMENT}/bin/activate

To activate environment (windows):
{NAME OF VIRTUAL ENVIRONMENT}/bin/activate

To deactivate:
deactivate



-----------------
To mirror virtual environment do the following:

While environemtn is activated, "pip freeze > requirements.txt"

On new machine, with environment activated:
pip install -r requirements.txt


