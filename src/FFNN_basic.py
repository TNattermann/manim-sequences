from manim import *
import numpy as np

class FeedforwardNNScene(Scene):
    def construct(self):
        # --- Input image (left)
        input_image = ImageMobject("media_input/mnist_digit.png").scale(1.8)
        input_image.to_edge(LEFT)
        self.add(input_image)

        # --- Neuron layout setup
        input_neurons = self.create_layer(10, x=-4)
        hidden1_neurons = self.create_layer(6, x=-1)
        hidden2_neurons = self.create_layer(6, x=2)
        output_neurons = self.create_layer(10, x=5)

        all_layers = [input_neurons, hidden1_neurons, hidden2_neurons, output_neurons]
        for layer in all_layers:
            self.add(layer)

        # --- Draw connections with initial weights
        edges = self.connect_layers(input_neurons, hidden1_neurons)
        edges += self.connect_layers(hidden1_neurons, hidden2_neurons)
        edges += self.connect_layers(hidden2_neurons, output_neurons)
        self.add(*edges)

        # --- Animate activation (forward pass)
        self.wait(0.5)
        self.animate_activation(input_neurons, color=BLUE)
        self.animate_activation(hidden1_neurons, color=YELLOW)
        self.animate_activation(hidden2_neurons, color=ORANGE)
        self.animate_activation(output_neurons, color=RED)

        # --- Predicted output highlight
        pred_idx = 3
        self.play(output_neurons[pred_idx].animate.set_fill(GREEN, opacity=1))

        # --- Loss breakdown per output
        loss_terms = VGroup()
        probs = np.random.dirichlet(np.ones(10))
        for i, p in enumerate(probs):
            label = MathTex(f"-{{1 if i==pred_idx else 0}} \, \log({p:.2f})").scale(0.5)
            label.next_to(output_neurons[i], RIGHT)
            if i == pred_idx:
                label.set_color(RED)
            loss_terms.add(label)
        self.play(LaggedStart(*[Write(lbl) for lbl in loss_terms], lag_ratio=0.1))
        self.wait(1)

        # --- Total loss formula
        total_loss = MathTex(f"Loss = -\log(p_{{{pred_idx}}}) = {-np.log(probs[pred_idx]):.2f}").scale(0.8)
        total_loss.to_corner(UR)
        self.play(Write(total_loss))
        self.wait(1.5)

        # --- Weight update effect (backprop)
        self.play(*[edge.animate.set_stroke(width=1.5, color=TEAL) for edge in edges])
        self.wait(1.5)

    def create_layer(self, num_neurons, x=0, y_spacing=0.7):
        neurons = VGroup()
        for i in range(num_neurons):
            neuron = Circle(radius=0.2, color=WHITE, fill_opacity=0.2)
            neuron.move_to(np.array([x, (i - num_neurons / 2) * -y_spacing, 0]))
            neurons.add(neuron)
        return neurons

    def connect_layers(self, layer1, layer2):
        edges = []
        for n1 in layer1:
            for n2 in layer2:
                line = Line(n1.get_center(), n2.get_center(), stroke_width=1, color=GRAY)
                edges.append(line)
        return edges

    def animate_activation(self, neurons, color=YELLOW):
        self.play(
            LaggedStart(*[n.animate.set_fill(color, opacity=0.8) for n in neurons], lag_ratio=0.05),
            run_time=0.5
        )