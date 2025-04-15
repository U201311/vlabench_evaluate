import json 
from collections import OrderedDict 


def get_name_list_by_class_name(first_class ,class_name, json_file):
    """
    Get the name list of a specific class from a JSON file.
    
    Args:
        first_class (str): The first level class in the JSON structure.
        class_name (str): The name of the class to search for.
        json_file (str): The path to the JSON file.
        
    Returns:
        list: A list of names associated with the specified class.
    """
    with open(json_file, 'r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    name_list = []
    class_name_dict = data[first_class][class_name]
    
    for item in class_name_dict:  # 直接遍历字典的 key
        name_list.append(item)
    
    return name_list



def get_obj_from_class(class_name, obj_class, asset_dict, obj_num):
    obj_class_dict = asset_dict[class_name]        
    obj_class_name_dict = obj_class_dict[obj_class]
    count = 0
    for key, value in obj_class_name_dict.items():
        if count == obj_num:
            print("key: ", key)
            print("value: ", value)
            return key, value
        count += 1