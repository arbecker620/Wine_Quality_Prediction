# Wine Quality Prediction


This project was an opportunity to start learning how to set up Github Actions and create a CICD framework for projects. There are a few different goals I am trying to achieve here:

1. Set up GitHub Actiions to do Model Training and upload the model to a Model Regsitry in my Azure ML Studio Subscription. 
2. Create a service in Fast API/Flask to host the model
3. Create a Dockerfile and Container for this model's service to go into Azure
4. Create a database using MongoDB where results can be stored in


[![](https://mermaid.ink/img/pako:eNp10M1qwzAMB_BXET63L5DDYPka7IOxNrBDnIOI1cQQ28GRaUvIu89pe0gP88XGv78k0Cxap0gkovM49lDl0kI8r_WvtgQ_AQfNV6g8aqttBzkyNrDfv6TzV6wb4EgDtez8cq9LV8vqA1rlDJTO08RwoC7ek_PNJlTUbx6VJsuQOjfxw7LVoJxvEyFz9qS7R-_ifypv9F4Xl5G8NmvTb0vNFj-2WJ3dE34-Ye-JGrEThrxBreJy5jUsBfdkSIokPhWdMAwshbRLjGJgd7zaViTsA-1EGBUy5RrjWs39c_kDC3R0qw?type=png)](https://mermaid.live/edit#pako:eNp10M1qwzAMB_BXET63L5DDYPka7IOxNrBDnIOI1cQQ28GRaUvIu89pe0gP88XGv78k0Cxap0gkovM49lDl0kI8r_WvtgQ_AQfNV6g8aqttBzkyNrDfv6TzV6wb4EgDtez8cq9LV8vqA1rlDJTO08RwoC7ek_PNJlTUbx6VJsuQOjfxw7LVoJxvEyFz9qS7R-_ifypv9F4Xl5G8NmvTb0vNFj-2WJ3dE34-Ye-JGrEThrxBreJy5jUsBfdkSIokPhWdMAwshbRLjGJgd7zaViTsA-1EGBUy5RrjWs39c_kDC3R0qw)

to run... 

docker build -t wine .

docker container run -it wine /bin/bash
