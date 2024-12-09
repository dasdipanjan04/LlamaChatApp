docker build --no-cache -t fastapi-llama-service:latest .
docker run --gpus all -e API_TOKEN="YOUR_API_TOKEN" -p 8000:8000 -p 8501:8501 fastapi-llama-service:latest
## Llama Based Chat App

# Instructions:

I have built the llama/mistral based chat app using the following resources:
* meta-llama/Llama-3.1-8B
* mistralai/Mistral-7B-v0.1
* huggingface
* FastAPI
* Docker

The needed python packages to run this project is mentioned in the requirements.txt file

### How to get access to the models:
In order to have the access to these gated models you need to login to Huggingface first, in case you 
don't have an account already, you need to create one at [Huggingface](https://huggingface.co/join) and verify your email.

Next step is to get the access to the gated repositories:

#### mistralai/Mistral-7B-v0.1:
You need to [Agree and access repository of Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1).
The access is granted instantly once accepted. Once accepted you will get to see the following:


![alt text](https://github.com/dasdipanjan04/LlamaChatApp/readme_resources/mistral_access.png?raw=true)

#### meta-llama/Llama-3.1-8B:
Getting access to [Meta-Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B) is a bit time-consuming.
It took around 15 minutes for me. First you need to fill out a basic form with you name and date of birth. You can use Affiliation as your current country,
for me, it was Germany.

The process will look like the following:

![alt text](https://github.com/dasdipanjan04/LlamaChatApp/readme_resources/meta-llama_access.png?raw=true)


![alt text](https://github.com/dasdipanjan04/LlamaChatApp/readme_resources/meta-llama_access_form.png?raw=true)

![alt text](https://github.com/dasdipanjan04/LlamaChatApp/readme_resources/meta-llama_pending_access.png?raw=true)

Once granted you will receive an email as well, and you will get to see the following:

![alt text](https://github.com/dasdipanjan04/LlamaChatApp/readme_resources/meta-llama_access_granted.png?raw=true)


