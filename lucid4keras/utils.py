import os
import tempfile
from keras.models import Model, load_model
from keras import activations

def prepare_model(model,layer_name="conv2d_5",linearize=True):
    '''
    input:
        model : a model built with keras.
        layer_name : a valid layer name within the model.
        linearize : will modify the specified layer from relu to linear. (actually the final layer of the generated intermediate model)
    return:
        modified keras model.
    '''
    def linearize_activations(input_model,layer_name):
        def search_inbound(layer_name):
            layer_list = []
            if "merge" in str(input_model.get_layer(layer_name)):
                #print("\tlayer",input_model.get_layer(layer_name).name,"is a merge layer. searching for connected layers: ")
                for layer in input_model.get_layer(layer_name)._inbound_nodes[0].inbound_layers:
                    search_inbound(layer.name)
            elif "pool" in str(input_model.get_layer(layer_name)):
                pass
            else:
                #print("\ttargeting layer:",input_model.get_layer(layer_name).name,input_model.get_layer(layer_name).activation)
                if input_model.get_layer(layer_name).activation == activations.linear:
                    print("already a linear layer")
                else:
                    print("\tlinearizing layer:",layer_name)
                    input_model.get_layer(layer_name).activation = activations.linear
            return 0

        if "merge" in str(input_model.get_layer(layer_name)) or "GlobalAveragePooling2D" in str(input_model.get_layer(layer_name)):
            print(layer_name,input_model.get_layer(layer_name),"is a merge layer. will linearize connected relu containing layers")
            #print("inbound layers are")
            #for layer in input_model.get_layer(layer_name)._inbound_nodes[0].inbound_layers:
            #    print("\t",layer.name)

            #print("will (recursively) search for layers connected to the specified layers until it hits a activation layer or conv2d layer having activations")
            _ = search_inbound(layer_name)
            model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + ".hdf5")
            input_model.save(model_path)
            input_model = load_model(model_path,compile=False)
            os.remove(model_path)
            return input_model
        else:
            #print("linearizing the specified layer:",layer_name,str(input_model.get_layer(layer_name)),input_model.get_layer(layer_name).activation)
            if input_model.get_layer(layer_name).activation == activations.linear:
                print("already a linear layer, return unmodified model")
                return input_model
            else:
                print("linearizing layer:",layer_name)
                input_model.get_layer(layer_name).activation = activations.linear
                model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + ".hdf5")
                input_model.save(model_path)
                input_model = load_model(model_path,compile=False)
                os.remove(model_path)
                return input_model
    

    model_intout = model.get_layer(layer_name).output
    int_model = Model(inputs=model.input,outputs=model_intout)

    if linearize:
        int_model = linearize_activations(int_model,layer_name)
    return int_model