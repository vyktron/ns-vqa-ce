import random
import json
import utils.utils as utils


CLEVR_COLORS = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow']
CLEVR_MATERIALS = ['rubber', 'metal']
CLEVR_SHAPES = ['cube', 'cylinder', 'sphere']
CLEVR_SIZES = ['large', 'small']


CLEVR_ANSWER_CANDIDATES = {
    'count': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'equal_color': ['yes', 'no'],
    'equal_integer': ['yes', 'no'],
    'equal_material': ['yes', 'no'],
    'equal_shape': ['yes', 'no'],
    'equal_size': ['yes', 'no'],
    'exist': ['yes', 'no'],
    'greater_than': ['yes', 'no'],
    'less_than': ['yes', 'no'],
    'query_color': ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow'],
    'query_material': ['metal', 'rubber'],
    'query_size': ['small', 'large'],
    'query_shape': ['cube', 'cylinder', 'sphere'],
    'same_color': ['yes', 'no'],
    'same_material': ['yes', 'no'],
    'same_size': ['yes', 'no'],
    'same_shape': ['yes', 'no']
}


class ClevrExecutor:
    """Symbolic program executor for CLEVR"""

    def __init__(self, val_scene_json, vocab_json):
        self.scenes = {
            'val': utils.load_scenes(val_scene_json)
        }
        self.vocab = utils.load_vocab(vocab_json)
        self.colors = CLEVR_COLORS
        self.materials = CLEVR_MATERIALS
        self.shapes = CLEVR_SHAPES
        self.sizes = CLEVR_SIZES
        self.answer_candidates = CLEVR_ANSWER_CANDIDATES

        self.modules = {}
        self._register_modules()
    
    def run(self, x, index, split, guess=False, debug=False, scene=None):
        assert self.modules and self.scenes, 'Must have scene annotations and define modules first'
        assert split == 'val'

        ans, temp = None, None

        # Find the length of the program sequence before the '<END>' token
        length = 0
        for k in range(len(x)):
            l = len(x) - k
            if self.vocab['program_idx_to_token'][x[l-1]] == '<END>':
                length = l
        if length == 0:
            return 'error'

        if scene is None:
            scene = self.scenes[split][index]
        self.exe_trace = []
        for j in range(length):
            i = length - 1 - j
            token = self.vocab['program_idx_to_token'][x[i]]
            if token == 'scene':
                if temp is not None:
                    ans = 'error'
                    break
                temp = ans
                ans = list(scene)
            elif token in self.modules:
                module = self.modules[token]
                if token.startswith('same') or token.startswith('relate'):
                    ans = module(ans, scene)
                else:
                    ans = module(ans, temp)
                if ans == 'error':
                    break
            self.exe_trace.append(ans)
            if debug:
                print(token)
                print('ans:')
                self._print_debug_message(ans)
                print('temp: ')
                self._print_debug_message(temp)
                print()
        ans = str(ans)

        if ans == 'error' and guess:
            final_module = self.vocab['program_idx_to_token'][x[0]]
            if final_module in self.answer_candidates:
                ans = random.choice(self.answer_candidates[final_module])
        return ans, scene

    def _print_debug_message(self, x):
        if type(x) == list:
            for o in x:
                print(self._object_info(o))
        elif type(x) == dict:
            print(self._object_info(x))
        else:
            print(x)

    def _object_info(self, obj):
        return '%s %s %s %s at %s' % (obj['size'], obj['color'], obj['material'], obj['shape'], str(obj['position']))
    
    def _register_modules(self):
        self.modules['count'] = self.count
        self.modules['equal_color'] = self.equal_color
        self.modules['equal_integer'] = self.equal_integer
        self.modules['equal_material'] = self.equal_material
        self.modules['equal_shape'] = self.equal_shape
        self.modules['equal_size'] = self.equal_size
        self.modules['exist'] = self.exist
        self.modules['filter_color[blue]'] = self.filter_blue
        self.modules['filter_color[brown]'] = self.filter_brown
        self.modules['filter_color[cyan]'] = self.filter_cyan
        self.modules['filter_color[gray]'] = self.filter_gray
        self.modules['filter_color[green]'] = self.filter_green
        self.modules['filter_color[purple]'] = self.filter_purple
        self.modules['filter_color[red]'] = self.filter_red
        self.modules['filter_color[yellow]'] = self.filter_yellow
        self.modules['filter_material[rubber]'] = self.filter_rubber
        self.modules['filter_material[metal]'] = self.filter_metal
        self.modules['filter_shape[cube]'] = self.filter_cube
        self.modules['filter_shape[cylinder]'] = self.filter_cylinder
        self.modules['filter_shape[sphere]'] = self.filter_sphere
        self.modules['filter_size[large]'] = self.filter_large
        self.modules['filter_size[small]'] = self.filter_small
        self.modules['greater_than'] = self.greater_than
        self.modules['less_than'] = self.less_than
        self.modules['intersect'] = self.intersect
        self.modules['query_color'] = self.query_color
        self.modules['query_material'] = self.query_material
        self.modules['query_shape'] = self.query_shape
        self.modules['query_size'] = self.query_size
        self.modules['relate[behind]'] = self.relate_behind
        self.modules['relate[front]'] = self.relate_front
        self.modules['relate[left]'] = self.relate_left
        self.modules['relate[right]'] = self.relate_right
        self.modules['same_color'] = self.same_color
        self.modules['same_material'] = self.same_material
        self.modules['same_shape'] = self.same_shape
        self.modules['same_size'] = self.same_size
        self.modules['union'] = self.union
        self.modules['unique'] = self.unique
        
    def count(self, scene, _):
        if type(scene) == list:
            return len(scene)
        return 'error'
    
    def equal_color(self, color1, color2):
        if type(color1) == str and color1 in self.colors and type(color2) == str and color2 in self.colors:
            if color1 == color2:
                return 'yes'
            else:
                return 'no'
        return 'error'
    
    def equal_integer(self, integer1, integer2):
        if type(integer1) == int and type(integer2) == int:
            if integer1 == integer2:
                return 'yes'
            else:
                return 'no'
        return 'error'
    
    def equal_material(self, material1, material2):
        if type(material1) == str and material1 in self.materials and type(material2) == str and material2 in self.materials:
            if material1 == material2:
                return 'yes'
            else:
                return 'no'
        return 'error'
    
    def equal_shape(self, shape1, shape2):
        if type(shape1) == str and shape1 in self.shapes and type(shape2) == str and shape2 in self.shapes:
            if shape1 == shape2:
                return 'yes'
            else:
                return 'no'
        return 'error'
    
    def equal_size(self, size1, size2):
        if type(size1) == str and size1 in self.sizes and type(size2) == str and size2 in self.sizes:
            if size1 == size2:
                return 'yes'
            else:
                return 'no'
        return 'error'
    
    def exist(self, scene, _):
        if type(scene) == list:
            if len(scene) != 0:
                return 'yes'
            else:
                return 'no'
        return 'error'
    
    def filter_blue(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'blue':
                    output.append(o)
            return output
        return 'error'
    
    def filter_brown(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'brown':
                    output.append(o)
            return output
        return 'error'
    
    def filter_cyan(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'cyan':
                    output.append(o)
            return output
        return 'error'
    
    def filter_gray(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'gray':
                    output.append(o)
            return output
        return 'error'
    
    def filter_green(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'green':
                    output.append(o)
            return output
        return 'error'
    
    def filter_purple(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'purple':
                    output.append(o)
            return output
        return 'error'
    
    def filter_red(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'red':
                    output.append(o)
            return output
        return 'error'
    
    def filter_yellow(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'yellow':
                    output.append(o)
            return output
        return 'error'
    
    def filter_rubber(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['material'] == 'rubber':
                    output.append(o)
            return output
        return 'error'
    
    def filter_metal(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['material'] == 'metal':
                    output.append(o)
            return output
        return 'error'
    
    def filter_cube(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['shape'] == 'cube':
                    output.append(o)
            return output
        return 'error'
    
    def filter_cylinder(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['shape'] == 'cylinder':
                    output.append(o)
            return output
        return 'error'
    
    def filter_sphere(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['shape'] == 'sphere':
                    output.append(o)
            return output
        return 'error'
    
    def filter_large(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['size'] == 'large':
                    output.append(o)
            return output
        return 'error'
    
    def filter_small(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['size'] == 'small':
                    output.append(o)
            return output
        return 'error'
    
    def greater_than(self, integer1, integer2):
        if type(integer1) == int and type(integer2) == int:
            if integer1 > integer2:
                return 'yes'
            else:
                return 'no'
        return 'error'
    
    def less_than(self, integer1, integer2):
        if type(integer1) == int and type(integer2) == int:
            if integer1 < integer2:
                return 'yes'
            else:
                return 'no'
        return 'error'
    
    def intersect(self, scene1, scene2):
        if type(scene1) == list and type(scene2) == list:
            output = []
            for o in scene1:
                if o in scene2:
                    output.append(o)
            return output
        return 'error'
    
    def query_color(self, obj, _):
        if type(obj) == dict and 'color' in obj:
            return obj['color']
        return 'error'
    
    def query_material(self, obj, _):
        if type(obj) == dict and 'material' in obj:
            return obj['material']
        return 'error'
    
    def query_shape(self, obj, _):
        if type(obj) == dict and 'shape' in obj:
            return obj['shape']
        return 'error'
    
    def query_size(self, obj, _):
        if type(obj) == dict and 'size' in obj:
            return obj['size']
        return 'error'
    
    def relate_behind(self, obj, scene):
        if type(obj) == dict and 'position' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['position'][1] < obj['position'][1]:
                    output.append(o)
            return output
        return 'error'
    
    def relate_front(self, obj, scene):
        if type(obj) == dict and 'position' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['position'][1] > obj['position'][1]:
                    output.append(o)
            return output
        return 'error'
    
    def relate_left(self, obj, scene):
        if type(obj) == dict and 'position' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['position'][0] < obj['position'][0]:
                    output.append(o)
            return output
        return 'error'
    
    def relate_right(self, obj, scene):
        if type(obj) == dict and 'position' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['position'][0] > obj['position'][0]:
                    output.append(o)
            return output
        return 'error'
    
    def same_color(self, obj, scene):
        if type(obj) == dict and 'color' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == obj['color'] and o['id'] != obj['id']:
                    output.append(o)
            return output
        return 'error'
    
    def same_material(self, obj, scene):
        if type(obj) == dict and 'material' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['material'] == obj['material'] and o['id'] != obj['id']:
                    output.append(o)
            return output
        return 'error'
    
    def same_shape(self, obj, scene):
        if type(obj) == dict and 'shape' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['shape'] == obj['shape'] and o['id'] != obj['id']:
                    output.append(o)
            return output
        return 'error'
    
    def same_size(self, obj, scene):
        if type(obj) == dict and 'size' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['size'] == obj['size'] and o['id'] != obj['id']:
                    output.append(o)
            return output
        return 'error'
    
    def union(self, scene1, scene2):
        if type(scene1) == list and type(scene2) == list:
            output = list(scene2)
            for o in scene1:
                if o not in scene2:
                    output.append(o)
            return output
        return 'error'
    
    def unique(self, scene, _):
        if type(scene) == list and len(scene) > 0:
            return scene[0]
        return 'error'
    
    # Contrastive Explanation methods
    def find_foil_answers(self, scene : list[dict], answer : str) -> list :
        """
        Find the foil answers for a given question type

        Parameters
        ----------
        scene : list[dict]
            The scene list of objects (dictionaries)
        answer : str
            The answer to the question (given by the model)
        
        Returns
        -------
        list
            The foil answers
        """

        # Get the answer candidates
        if answer == "yes" :
            candidates = ["no"]
        elif answer == "no" :
            candidates = ["yes"]
        elif answer in self.colors :
            # Get the colors that are not the answer
            candidates = [c for c in self.colors if c != answer]
        elif answer in self.materials :
            # Get the materials that are not the answer
            candidates = [m for m in self.materials if m != answer]
        elif answer in self.shapes :
            # Get the shapes that are not the answer
            candidates = [s for s in self.shapes if s != answer]
        elif answer in self.sizes :
            # Get the sizes that are not the answer
            candidates = [s for s in self.sizes if s != answer]
        elif answer in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"] :
            # Get the integers that are not the answer
            candidates = [str(i) for i in range(11) if str(i) != answer]

        return candidates
    
    # Modification of the scene

    def add_object(self, scene : list[dict], cost=0) -> list :
        """
        Add an object to the scene

        Parameters
        ----------
        scene : list[dict]
            The scene list of objects (dictionaries)

        Returns
        -------
        list
            The new scene
        """

        cost += 12

        # Deep copy the scene
        scene_ = json.loads(json.dumps(scene))
        # Construct the new object
        obj = {
            "id": str(scene_[0]["id"].split("-")[1]) + "-" + str(len(scene_)),
            # Random position
            "position": [random.uniform(-3, 3), random.uniform(-3, 3), 0.35],
            "color": random.choice(self.colors),
            "material": random.choice(self.materials),
            "shape": random.choice(self.shapes),
            "size": random.choice(self.sizes),
            "confidence": 1
        }

        if obj["size"] == "large" :
            obj["position"][2] = 0.7
        
        desc = "add object " + str(len(scene_))

        # Add the object to the scene
        scene_.append(obj)     

        return scene_, cost, desc
    
    def remove_object(self, scene : list[dict], cost=0) -> list :
        """
        Remove an object from the scene

        Parameters
        ----------
        scene : list[dict]
            The scene list of objects (dictionaries)
        
        Returns
        -------
        list
            The new scene
        """

        cost += 12

        # Deep copy the scene
        scene_ = json.loads(json.dumps(scene))
        # Get a random index
        index = random.randrange(len(scene_))
        # Remove the object
        scene_.pop(index)

        desc = "remove_object " + str(index)

        return scene_, cost, desc
    
    def change_shape(self, scene : list[dict], cost=0) -> list :
        """
        Change the shape of an object in the scene

        Parameters
        ----------
        scene : list[dict]
            The scene list of objects (dictionaries)
        object_id : int
            The id of the object to change the shape
        
        Returns
        -------
        list
            The new scene
        """

        cost += 11

        # Deep copy the scene
        scene_ = json.loads(json.dumps(scene))
        # Get a random index
        index = random.randrange(len(scene_))
        # Get the object to change the shape
        obj = scene_[index]

        prev_shape = obj["shape"]
        # Change the shape
        obj["shape"] = random.choice([s for s in self.shapes if s != obj["shape"]])

        desc = "change_shape " + str(index) + " : " + prev_shape + " -> " + obj["shape"]

        return scene_, cost, desc
    
    def change_color(self, scene : list[dict], cost=0) -> list :
        """
        Change the color of an object in the scene

        Parameters
        ----------
        scene : list[dict]
            The scene list of objects (dictionaries)
        object_id : int
            The id of the object to change the color
        
        Returns
        -------
        list
            The new scene
        """

        cost += 11

        # Deep copy the scene
        scene_ = json.loads(json.dumps(scene))
        # Get a random index
        index = random.randrange(len(scene_))
        # Get the object to change the color
        obj = scene_[index]

        prev_color = obj["color"]
        # Change the color
        obj["color"] = random.choice([c for c in self.colors if c != obj["color"]])

        desc = "change_color " + str(index) + " : " + prev_color + " -> " + obj["color"]

        return scene_, cost, desc
    
    def change_material(self, scene : list[dict], cost=0) -> list :
        """
        Change the material of an object in the scene

        Parameters
        ----------
        scene : list[dict]
            The scene list of objects (dictionaries)
        object_id : int
            The id of the object to change the material
        
        Returns
        -------
        list
            The new scene
        """

        cost += 11

        # Deep copy the scene
        scene_ = json.loads(json.dumps(scene))
        # Get a random index
        index = random.randrange(len(scene_))
        # Get the object to change the material
        obj = scene_[index]

        prev_material = obj["material"]
        # Change the material
        obj["material"] = random.choice([m for m in self.materials if m != obj["material"]])

        desc = "change_material " + str(index) + " : " + prev_material + " -> " + obj["material"]

        return scene_, cost, desc
    
    def change_size(self, scene : list[dict], cost=0) -> list :
        """
        Change the size of an object in the scene

        Parameters
        ----------
        scene : list[dict]
            The scene list of objects (dictionaries)
        object_id : int
            The id of the object to change the size
        
        Returns
        -------
        list
            The new scene
        """

        cost += 11

        # Deep copy the scene
        scene_ = json.loads(json.dumps(scene))
        # Get a random index
        index = random.randrange(len(scene_))
        # Get the object to change the size
        obj = scene_[index]

        prev_size = obj["size"]
        # Change the size
        obj["size"] = random.choice([s for s in self.sizes if s != obj["size"]])
        if obj["size"] == "large" :
            obj["position"][2] = 0.7
        elif obj["size"] == "small" :
            obj["position"][2] = 0.35

        desc = "change_size " + str(index) + " : " + prev_size + " -> " + obj["size"]

        return scene_, cost, desc

    def move_object(self, scene : list[dict], cost=0) -> list :
        """
        Move an object in the scene to a new position

        Parameters
        ----------
        scene : list[dict]
            The scene list of objects (dictionaries)

        Returns
        -------
        list
            The new scene
        """

        cost += 11

        # Deep copy the scene
        scene_ = json.loads(json.dumps(scene))
        # Get a random index
        index = random.randrange(len(scene_))
        # Get the object to move
        obj = scene_[index]
        # Choose a random direction
        direction = random.choice(["up", "down", "left", "right"])
        # Move the object in the chosen direction (by 0.5)
        if direction == "up" :
            obj["position"][1] += 0.5
        elif direction == "down" :
            obj["position"][1] -= 0.5
        elif direction == "left" :
            obj["position"][0] -= 0.5
        elif direction == "right" :
            obj["position"][0] += 0.5

        desc = "move " + direction + " " + str(index)

        return scene_, cost, desc
    
    def modify_scene(self, scene : list[dict], cost=0) -> list :
        """
        Modify the scene by adding, removing, moving or changing an object

        Parameters
        ----------
        scene : list[dict]
            The scene list of objects (dictionaries)
        
        Returns
        -------
        list
            The new scene
        """

        # Deep copy the scene
        scene_ = json.loads(json.dumps(scene))
        # Choose a random modification
        modification = random.choice(["add", "remove", "move", "change_size", "change_shape", "change_color", "change_material"])
        # Modify the scene
        if modification == "add" :
            scene_, cost, desc = self.add_object(scene_, cost)
        elif modification == "remove" :
            scene_, cost, desc = self.remove_object(scene_, cost)
        elif modification == "move" :
            scene_, cost, desc = self.move_object(scene_, cost)
        elif modification == "change_size" :
            scene_, cost, desc = self.change_size(scene_, cost)
        elif modification == "change_shape" :
            scene_, cost, desc = self.change_shape(scene_, cost)
        elif modification == "change_color" :
            scene_, cost, desc = self.change_color(scene_, cost)
        else :
            scene_, cost, desc = self.change_material(scene_, cost)

        return scene_, cost, desc
    