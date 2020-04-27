import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
import numpy as np
import time
import PySimpleGUI as sg

# display images in notebook
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# declare global variables
data_dir = 'datasets'
mod_dir = 'models'
onnx_model = '{}/model.onnx'.format(mod_dir)
onnx_labels = '{}/labels.txt'.format(mod_dir)

def run_model():
    # Run the model on the backend
    session = onnxruntime.InferenceSession(onnx_model, None)

    # get the name of the first input of the model
    input_name = session.get_inputs()[0].name  
    
    return input_name, session

def load_labels(path):
    # read from a .txt file
    list_data = []
    with open(path) as f:
        data = f.read().split('\n')  
        #print("data: %s" %(data))
        list_data.extend(data)
    return np.asarray(list_data)

def postprocess(results):
    pred_result = results[0]
    #print("pred_result:", pred_result)
    list_prob = np.delete(results, 0, 0)
    dict_prob = list_prob.reshape(-1)[0]
    #for k in dict_prob.keys():
    #    print("Probablities for %s is %.4f" %(k, dict_prob[k]))
    return np.array(list(dict_prob.values()))

def display_bird(img_file):
    fig = plt.figure(figsize=(5,5))
    plt.rcParams.update({'font.size': 12})
 
    #img_file = sg.PopupGetFile('Please enter a file name')
    #print("Image file:", img_file)
    
    #image = Image.open(img_file).resize((499, 499), Image.LANCZOS)
    image = Image.open(img_file).resize((224, 224), Image.LANCZOS)

    #print("Image size: ", image.size)
    plt.axis('off')
    plt.title(img_file)
    plt.imshow(image)
    fig_dir = 'static'
    fig1_name = 'plots/bird.png'
    fig.savefig(fig_dir+'/'+fig1_name, bbox_inches='tight')
    plt.close()
    
    return fig1_name

def plot_bar_prob(list_x, list_y, g_title, x_label):
    """ To plot bar chart from a list """
    
    #print("Plotting bar chart for prediction probabilities...")
    
    fig = plt.figure(figsize=(6,6))
    plt.rcParams.update({'font.size': 12})
    x_labels = list_x
    formatted_list_y = [ '%.4f' % elem for elem in list_y ]
    #print("y_list:", formatted_list_y)
    #x_pos = np.arange(len(list_x))  #the label locations
    ax=sns.barplot(x=list_x, y=list_y)
    ax.set_title(g_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("probabilities")
    ax.set_xticklabels(x_labels, rotation=40, ha='right')
    #plt.tight_layout()
    i = 0
    for i in range(len(formatted_list_y)):
        plt.text(i, list_y[i], str(formatted_list_y[i]), ha='center')
        i += 1
    fig_dir = 'static'
    fig_name = 'plots/prob.png'
    fig.savefig(fig_dir+'/'+fig_name, bbox_inches='tight')
    plt.close()
    
    return fig_name
        
def classify_bird(img_file):
#def classify_bird():
    """ To classify the bird """
    
    input_name, session = run_model()
    #print('Input Name:', input_name)
    labels = load_labels(onnx_labels)
    #print("Labels:", type(labels), labels.shape, len(labels), labels)
    
    #img_file = sg.PopupGetFile('Please enter a file name')
    #print("Image size: ", image.size)
    print("Image: ", img_file)
    
    # image normalization
    image = Image.open(img_file).resize((224, 224), Image.LANCZOS)
    image_data = np.array(image).transpose(2, 0, 1)
    norm_img_data = cv2.normalize(image_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    input_data = norm_img_data.reshape(1, 3, 224, 224)
        
    start = time.time()
    raw_result = session.run([], {input_name: input_data})
    end = time.time()
    #print(raw_result)
    list_results = postprocess(raw_result)
    #print("res of postprocess:", len(list_results), list_results)
    
    sort_idx = np.flip(np.squeeze(np.argsort(list_results)))
    list_labels = labels[sort_idx[:5]]
    list_prob = list_results[sort_idx[:5]]
    
    fig2_name = plot_bar_prob(list_labels, list_prob, 'Bird classification', 'Top 5')
    
    inference_time = np.round((end - start) * 1000, 2)
    print('Inference time: ' + str(inference_time) + " ms")
    
    return fig2_name

#fig2_name = classify_bird()
#print("Figures:", fig2_name)





