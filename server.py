from fastapi import FastAPI, Request
import uvicorn
import os
import helper

app=FastAPI()



@app.post("/api/face_recognition_url/")
async def face_recognition_api(info: Request):
    data= await info.json()
    id=data['id']
    image=data['image']
    test_image=data['test_image']
    image=helper.get_image_from_url(image)
    test_image=helper.get_image_from_url(test_image)
    result,distance=helper.predict(image,test_image)
    result=str(result[0])
    distance=str(distance[0])
    return {'id':id,'result':result,'distance':distance}

@app.post("/api/face_recognition_path/")
async def face_recognition_api(info: Request):
    data= await info.json()
    id=data['id']
    image=data['image']
    test_image=data['test_image']
    image=helper.get_image_from_path(image)
    test_image=helper.get_image_from_path(test_image)
    result,distance=helper.predict(image,test_image)
    result=str(result[0])
    distance=str(distance[0])
    return {'id':id,'result':result,'distance':distance}

if __name__=='__main__':
    ## uncomment to run locally
    #uvicorn.run(app)

    ## uncomment to run on heroku
    uvicorn.run("server:app", host='0.0.0.0', port=(int)(os.environ.get('PORT', 5001)))
