import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from PIL import Image
import torch
from pydantic import BaseModel
from zipfile import ZipFile
import os
import json
import numpy as np
import faiss

##############################################################
# Import image and text processors                           #
##############################################################

from resnet_classifier import TransferLearning
from clean_images import resize_image, image_processor
from dataset import import_tabular_data


def get_image_decoder():
    with open('encoder_decoder.json') as f:
        load_encoder_decoder = json.load(f)
    decoder_str = load_encoder_decoder['decoder']
    decoder = {int(k): v for k, v in decoder_str.items()}
    return decoder

# class TextClassifier(nn.Module):
#     def __init__(self,
#                  decoder: dict = None):
#         super(TextClassifier, self).__init__()
#         pass

# ##############################################################
# # TODO                                                       #
# # Populate the __init__ method, so that it contains the same #
# # structure as the model used to train the text model        #
# ##############################################################
        
#         self.decoder = decoder
#     def forward(self, text):
#         x = self.main(text)
#         return x

#     def predict(self, text):
#         with torch.no_grad():
#             x = self.forward(text)
#             return x
    
#     def predict_proba(self, text):
#         with torch.no_grad():
#             pass


#     def predict_classes(self, text):
#         with torch.no_grad():
#             pass

class ImageClassifier(TransferLearning):
    def __init__(self,
                 decoder: dict = None):
        super().__init__()        
        self.decoder = decoder

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

    def predict_proba(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return torch.softmax(x, dim=1).tolist()[0]

    def predict_classes(self, image):
        with torch.no_grad():
            x = self.forward(image)
            int_prediction = torch.argmax(x, dim=1).item() # needs to be an int to index decoder
            return self.decoder[int_prediction]


class FaissSearchIndex():
    def __init__(self) -> None:
        self.image_embeddings = np.load('./image_embeddings.npy')
        self.d = self.image_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.d)
        self.index.add(self.image_embeddings)
        data_df = import_tabular_data()
        self.decoder = dict(zip(list(data_df.image_id.values), list(data_df.index)))
        self.encoder = dict(zip(list(data_df.index), list(data_df.image_id.values)))
    
    def get_similarity_index(self, image_id, k=2):
        idx = self.decoder[image_id]
        xq = self.image_embeddings[idx,:]
        xq = xq[np.newaxis, :]
        D, I = self.index.search(xq, k)
        if k == 2:
            return self.encoder[I[0][-1]]
        else:
            return I[0]
    
    def get_encoded_image_id(self, idx):
        return self.encoder[idx]


# class CombinedModel(nn.Module):
#     def __init__(self,
#                  decoder: list = None):
#         super(CombinedModel, self).__init__()
# ##############################################################
# # TODO                                                       #
# # Populate the __init__ method, so that it contains the same #
# # structure as the model used to train the combined model    #
# ##############################################################
        
#         self.decoder = decoder

#     def forward(self, image_features, text_features):
#         pass

#     def predict(self, image_features, text_features):
#         with torch.no_grad():
#             combined_features = self.forward(image_features, text_features)
#             return combined_features
    
#     def predict_proba(self, image_features, text_features):
#         with torch.no_grad():
#             pass

#     def predict_classes(self, image_features, text_features):
#         with torch.no_grad():
#             pass



# Useful for one of the methods in the API
class TextItem(BaseModel):
    text: str


# try:
##############################################################
# TODO                                                       #
# Load the text model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the text model    #
# used for training it, and then load the weights in it.     #
# Also, load the decoder dictionary                          #
##############################################################
#     pass
# except:
#     raise OSError("No Text model found.")

try:
##############################################################
# Load the image model. Initialize a class that inherits     #
# from nn.Module, and has the same structure as the image    #
# model used for training, and load the weights in it.       #
##############################################################
    image_decoder = get_image_decoder()
    image_classifier = ImageClassifier(decoder=image_decoder)
    image_classifier.load_state_dict(torch.load('resnet_model_weights.pth'))
except:
    raise OSError("No Image model found.")

try:
    faiss_mdl = FaissSearchIndex()
except:
    raise OSError('No image embeddings found, check that they are in the right location')

# try:
##############################################################
# TODO                                                       #
# Load the text model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the text model    #
# used for training, then load the weights in it.            #
##############################################################
#     pass
# except:
    # raise OSError("No Combined model found.")

# try:
##############################################################
# TODO                                                       #
# Initialize the text processor for processing the text that #
# users will send to tthe API.                               #
# Make sure that the max_length is the same as when the      # 
# model was trained.                                         #
##############################################################
#     pass
# except:
#     raise OSError("No Text processor found.")

app = FastAPI()
print("Starting server")

@app.get("/")
def read_root():
    return {"Dan's app": "Welcome to the FB Marketplace Ranking App" }

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

# @app.post('/predict/text')
# def predict_text(text: TextItem):
  
#     ##############################################################
#     # TODO                                                       #
#     # Process the input and use it as input for the text model   #
#     # text.text is the text that the user sends to the API       #
#     # Apply the corresponding methods to compute the category    #
#     # and the probabilities                                      #
#     ##############################################################

#     return JSONResponse(content={
#         "Category": "", # Return the category
#         "Probabilities": "" # Return a list or dict of probabilities
#             })
  
  
@app.post('/predict/image')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)
    resized_image = resize_image(pil_image, 64)
    tensor_image = image_processor(resized_image)

    predicted_class = image_classifier.predict_classes(tensor_image)
    
    classes = list(image_decoder.values())
    predicted_probabilities = image_classifier.predict_proba(tensor_image)
    probabilities_dict = dict(zip(classes, predicted_probabilities))

    probabilities_dict = {k:round(v,4) for k, v in probabilities_dict.items()}
    sorted_probabilities_dict = dict(sorted(probabilities_dict.items(), key=lambda x:x[1], reverse=True))

    return JSONResponse(content={
    "Category": predicted_class, # Returns the category
    "Probabilities": sorted_probabilities_dict # Returns a dict of probabilities
        })
  
# @app.post('/predict/combined')
# def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
#     print(text)
#     pil_image = Image.open(image.file)
    
#     ##############################################################
#     # TODO                                                       #
#     # Process the input and use it as input for the image model  #
#     # image.file is the image that the user sends to the API     #
#     # Apply the corresponding methods to compute the category    #
#     # and the probabilities                                      #
#     ##############################################################

#     return JSONResponse(content={
#     "Category": "", # Returns the category
#     "Probabilities": "" # Returns a dict of probabilities
#         })


@app.post('/predict/suggest_similar_images')
def faiss_similar_images(image: UploadFile = File(...)):
    
    image_id = image.filename[:-4]
    similar_image_id = faiss_mdl.get_similarity_index(image_id)
    output_filename = similar_image_id + '.jpg'

    output_path = './raw_images/' + output_filename
    if not os.path.exists(output_path):
        zipfile_path = 'images/' + output_filename
        if not os.path.exists('./raw_images'):
            os.mkdir('./raw_images')
        with ZipFile('./images_fb.zip') as myzip:
            myzip.extract(zipfile_path, './raw_images')
        os.replace(f'./raw_images/images/{output_filename}', output_path)
        os.rmdir('./raw_images/images')


    return FileResponse(path=output_path)
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)