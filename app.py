from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSeq2SeqLM
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer

app = Flask(__name__)

languages = {'assamese':'asm_Beng','kannada':'kan_Knda','Tamil':'tam_Taml','telugu':'tel_Telu','malayalam':'mal_Mlym','odia':'ory_Orya','punjabi':'pan_Guru',
             'bengali':'ben_Beng','gujarati':'guj_Gujr','urdu':'urd_Arab','marathi':'mar_Deva','hindi':'hin_Deva','english':'eng_Latn'
            }

@app.route('/translate', methods=['POST'])
def post_example():
    try:
        # Assuming the incoming data is in JSON format
        data = request.get_json()
        src_lang = data['src_lang'].lower()
        tar_lang = data['tar_lang'].lower()
        sentence = data['para']

        if src_lang=='english':
            tokenizer = IndicTransTokenizer(direction="en-indic")
            ip = IndicProcessor(inference=True)
            model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)
            paragraph = [sentence]
            batch = ip.preprocess_batch(paragraph, src_lang=languages[src_lang], tgt_lang=languages[tar_lang])
            batch = tokenizer(batch, src=True, return_tensors="pt")

            with torch.inference_mode():
                outputs = model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)
            outputs = tokenizer.batch_decode(outputs, src=False)
            outputs = ip.postprocess_batch(outputs, lang=languages[tar_lang])
            return jsonify({'translated':outputs[0]})
        
        elif tar_lang=='english':
            tokenizer = IndicTransTokenizer(direction="indic-en")
            ip = IndicProcessor(inference=True)
            model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-en-1B", trust_remote_code=True)
            paragraph = [sentence]
            batch = ip.preprocess_batch(paragraph, src_lang=languages[src_lang], tgt_lang=languages[tar_lang])
            batch = tokenizer(batch, src=True, return_tensors="pt")

            with torch.inference_mode():
                outputs = model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)
            outputs = tokenizer.batch_decode(outputs, src=False)
            outputs = ip.postprocess_batch(outputs, lang=languages[tar_lang])
            return jsonify({'translated':outputs[0]})
        
        else:
            tokenizer = IndicTransTokenizer(direction="indic-indic")
            ip = IndicProcessor(inference=True)
            model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-indic-1B", trust_remote_code=True)
            paragraph = [sentence]
            batch = ip.preprocess_batch(paragraph, src_lang=languages[src_lang], tgt_lang=languages[tar_lang])
            batch = tokenizer(batch, src=True, return_tensors="pt")

            with torch.inference_mode():
                outputs = model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)
            outputs = tokenizer.batch_decode(outputs, src=False)
            outputs = ip.postprocess_batch(outputs, lang=languages[tar_lang])
            return jsonify({'translated':outputs[0]})
        

        # Sending a response

    except Exception as e:
        # Handling exceptions, you may want to customize this part based on your needs
        error_response = {'status': 'error', 'message': str(e)}
        return jsonify(error_response), 400  # Return a 400 status code for bad requests

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)
