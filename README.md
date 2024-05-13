# LICENSE PLATE DETECTION 🚘


Created a fastapi API endpoint to detect license plate numbers. To try it out simply run 

1. `pip install -r requirements.txt `
2. `uvicorn main:app --reload`
3. Go to the `\docs` page to test it out by upload the images from the test_images folder (or your own images !)


**Additional tips:**
1. If you run into issues when loading the model, **ahem** more specifically this one:`ImportError: libGL.so.1: cannot open shared object file: No such file or directory`

Then you can try doing this: 
- as the root user run the below command (`su root` to swap into root)
- `apt-get update && apt-get install libgl1`


*Credits to Jun Ming for providing the model training code*
