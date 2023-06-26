# from flask import *
# import json, time

# app = Flask(__name__)

# @app.route('/', methods=['GET'])
# def home_page():
#     data_set = {'Page': 'Home', 'Message': "Hello world", 'Timestamp': time.time()}
#     json_dump = json.dumps(data_set)

#     return json_dump

# # @app.route('/user/', methods=['GET'])
# # def home_page():
# #     user_query = str(request.args.get('user')) #/user/?user=USER_NAME

# #     data_set = {'Page': 'Request', f'Sucessfully got the request for {user_query}': "Hello world", 'Timestamp': time.time()}
# #     json_dump = json.dumps(data_set)

# #     return json_dump


# if __name__ == '__main__':
#     app.run(port=7777)


# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# # Load pre-trained model and tokenizer
# model_name = "gpt2"  # or specify a different model such as "gpt2-medium"
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# # Set the model to generate mode
# model.eval()

# # Generate text
# prompt = "Once upon a time"
# input_ids = tokenizer.encode(prompt, return_tensors="pt")

# # Generate multiple sequences
# num_sequences = 5
# max_length = 50

# for _ in range(num_sequences):
#     # Generate a sequence
#     output = model.generate(
#         input_ids=input_ids,
#         max_length=max_length,
#         num_return_sequences=1,
#         do_sample=True,
#         temperature=0.7,  # Controls the randomness of the generated text. Higher values produce more random output.
#     )
    
#     # Decode the generated sequence
#     generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
#     print("Generated Text:", generated_text)
#     print("-" * 50)


import tensorflow as tf
from tensorflow.keras import layers

# Define the model architecture
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_dim,)),  # Input layer with 64 neurons and ReLU activation
    layers.Dense(32, activation='relu'),  # Hidden layer with 32 neurons and ReLU activation
    layers.Dense(num_classes, activation='softmax')  # Output layer with softmax activation for multi-class classification
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Print the model summary
model.summary()
