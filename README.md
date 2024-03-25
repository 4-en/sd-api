# sd-api
Simple api for running hf diffusion models.

### Setup
- Create new Environment (optional)
  - python -m venv venv && source venv/bin/activate
- Install requirements
  - pip install -r requirements.txt
 
### Usage
Run either server or client
#### Server
Runs a diffuser model on a server using fastapi
```
python sd_api.py
```
#### Client
Uses a server to convert an input video and show it in a window
```
python sd_client.py --option VALUE ...
```
Use -h to see options
#### Example Request
Check out example_request.py to see how to process one or multiple images.



