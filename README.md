# Run project locally:


# Deploy gcloud run

## Step 1:

gcloud auth login

## Step 2:
gcloud init

## Step 3:
gcloud config set project project name


## Step 4:
gcloud builds submit --tag gcr.io/tactile-education-services-pvt/image-app

## Step 5:
gcloud run deploy --image gcr.io/tactile-education-services-pvt/image-app



python 3.7 location:

<!-- c:\users\rites\appdata\local\programs\python\python37\python.exe -m pip install --upgrade pip -->