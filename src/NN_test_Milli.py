from manim import *
from manim_ml.neural_network import FeedForwardLayer, NeuralNetwork


class PlainNN(Scene):
    def construct(self):
        # build NN layers
        input_layer = FeedForwardLayer(6, node_spacing=0.7)
        hidden_layer = FeedForwardLayer(5, node_spacing=0.7)
        hidden_layer_2 = FeedForwardLayer(5, node_spacing=0.7)
        output_layer = FeedForwardLayer(10, node_spacing=0.7)

        #build connections between layers
        nn = NeuralNetwork([input_layer, hidden_layer, hidden_layer_2, output_layer], layer_spacing=1.5)
        nn.move_to(ORIGIN)

        self.add(nn)

        # make neurons accessible
        input_neurons = input_layer.node_group.submobjects
        hidden_neurons = hidden_layer.node_group.submobjects
        hidden_neurons_2 = hidden_layer_2.node_group.submobjects
        output_neurons = output_layer.node_group.submobjects

        # manually create connections to access them later TODO
        connections = []
        for i, start_neuron in enumerate(input_neurons):
            for j, end_neuron in enumerate(hidden_neurons):
                line = Line(start_neuron.get_center(), end_neuron.get_center(), stroke_opacity=0.4)
                connections.append((start_neuron, end_neuron, line))

        # find middle to add ... for neurons that are not displayed
        top_neuron = input_neurons[2]
        bottom_neuron = input_neurons[3]
        mid_point = (top_neuron.get_center() + bottom_neuron.get_center()) / 2

        # add Text with dots
        dots = Tex(r"\vdots", font_size=30)
        dots.move_to(mid_point)
        self.add(dots)
        # makes sure the dots are in the front (visible)
        dots.set_z_index(100)

        # add labels for input layer
        input_labels = ["784", "783", "782", "3", "2", "1"]
        for neuron, label in zip(input_neurons, input_labels):
            text = Text(label, font_size=20)
            text.next_to(neuron, LEFT, buff=0.2)
            self.add(text)

        # add labels for output layer
        label_neuron_pairs = []
        labels = ["9", "8", "7", "6", "5", "4", "3", "2", "1", "0"]
        for neuron, label in zip(output_neurons, labels):
            text = Text(label, font_size=20)
            text.next_to(neuron, RIGHT, buff=0.2)
            self.add(text)
            label_neuron_pairs.append(VGroup(neuron, text))

        # fill some neurons on hidden layer 1 and according connections TODO
        index_list = [0, 1, 3]
        for pos in index_list:
            current_neuron = hidden_neurons[pos]
            current_neuron.set_fill(WHITE, opacity=0.9)
            for neuron in input_neurons:
                for start, end, line in nn.connections:
                    if start == neuron and end == current_neuron:
                        line.set_color(RED, opacity=0.9)


        # fill neurons on hidden layer 2 TODO like on hidden layer 1
        index_list = [1,2]
        for pos in index_list:
            hidden_neurons_2[pos].set_fill(WHITE, opacity=0.9)

        # fill one output neuron 
        index = 5
        target_neuron = output_neurons[index]
        target_neuron.set_fill(WHITE, opacity=0.9)

        # add rectangle around
        highlight_group = label_neuron_pairs[index]
        box = SurroundingRectangle(highlight_group, color=BLUE, buff=0.15)
        box.set_stroke(width=4)

        self.add(box)
        box.set_z_index(100)


        # create Output
        # self.play(Create(nn))
        # print(nn)
        