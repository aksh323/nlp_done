from flask import Flask, request, jsonify

import torch

from transformers import AutoModelForSeq2SeqLM

from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import os

app = Flask(__name__)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

languages = {

    'assamese': 'asm_Beng', 'kannada': 'kan_Knda', 'Tamil': 'tam_Taml', 'telugu': 'tel_Telu',

    'malayalam': 'mal_Mlym', 'odia': 'ory_Orya', 'punjabi': 'pan_Guru',

    'bengali': 'ben_Beng', 'gujarati': 'guj_Gujr', 'urdu': 'urd_Arab',

    'marathi': 'mar_Deva', 'hindi': 'hin_Deva', 'english': 'eng_Latn'

}

#cuda = torch.device('cuda')     # Default CUDA device
#cuda0 = torch.device('cuda')
#cuda2 = torch.device('cuda:2') 
#device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B",trust_remote_code=True)
model.to('cuda:0')
#model1 = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-en-1B",trust_remote_code=True)

#model2 = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-indic-1B",trust_remote_code=True)


# if torch.cuda.is_available():
#     print('yes')
#     model = model.to(cuda0)
#     torch.cuda.empty_cache()#
#     #model1 = model1.to(cuda0)  #
#     #torch.cuda.empty_cache()
#     #model2 = model2.to(cuda2)  #
#     #torch.cuda.empty_cache()

# device = torch.device("cuda")
# model.cuda()
def translate(sentence, src_lang, tar_lang):
    torch.cuda.empty_cache()

    device = 'cuda:0'  # Define the device

    print(f"Translating sentence: {sentence} from {src_lang} to {tar_lang}")

    if src_lang == 'english':
        tokenizer = IndicTransTokenizer(direction="en-indic")
    elif tar_lang == 'english':
        tokenizer = IndicTransTokenizer(direction="indic-en")
    else:
        tokenizer = IndicTransTokenizer(direction="indic-indic")

    ip = IndicProcessor(inference=True)
    paragraph = [sentence]
    batch = ip.preprocess_batch(paragraph, src_lang=languages[src_lang], tgt_lang=languages[tar_lang])
    batch = tokenizer(batch, src=True, return_tensors="pt")

    with torch.inference_mode():
        # Move input tensor to the desired device
        for k, v in batch.items():
            batch[k] = v.to(device)

        outputs = model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)

    outputs = tokenizer.batch_decode(outputs, src=False)
    outputs = ip.postprocess_batch(outputs, lang=languages[tar_lang])

    print(f"Translated output: {outputs[0]}")
    return outputs[0]



@app.route('/translate', methods=['POST'])

def post_example():

    # try:

        # Assuming the incoming data is in JSON format

    data = request.get_json()
    print(f"Incoming data: {data}")

    src_lang = data['src_lang'].lower()

    tar_lang = data['tar_lang'].lower()

    key_values = data['key_values']

    translated_data =  [{'title':translate(data['key_values'][0]['title'],src_lang,tar_lang)},{'summary':translate(data['key_values'][1]['summary'],src_lang,tar_lang)}]

    print(f"Translated data: {translated_data}")

    return jsonify(translated_data)


#     except Exception as e:

#         print(f"An error occurred: {str(e)}")

#         error_response = {'status': 'error', 'message': str(e)}

#         return jsonify(error_response), 400  # Return a 400 status code for bad requests


@app.route('/',methods=['GET'])

def hello():
    data=request.get_json()
    print (data)

    return jsonify({'message':'working'})


if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=8000)




