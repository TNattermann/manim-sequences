from manim import *
import numpy as np
import random

class FullPassNNScene(Scene):
    def construct(self):

        # Hyperparams
        input_size = 784
        visible_input = 5
        hidden1_size = 16
        hidden2_size = 16
        output_size = 10

        # Dummy data
        pred_output = np.random.rand(output_size)
        pred_output /= pred_output.sum()
        label = 3
        losses = [-int(i == label) * np.log(p + 1e-8) for i, p in enumerate(pred_output)]
        total_loss = sum(losses)


        # --- Input image (left)
        input_image = ImageMobject("media_input/mnist_digit.png").scale(1.6)
        input_image.to_edge(LEFT)

        # --- Create layers
        input_layer, input_neurons = self.create_input_layer(total=input_size, visible=visible_input, spacing=0.3)
        hidden1 = self.create_layer(hidden1_size)
        hidden2 = self.create_layer(hidden2_size)
        output_layer = self.create_layer(output_size, spacing = 0.6)

        # --- Arrange layers and position next to input image
        all_layers = VGroup(input_layer, hidden1, hidden2, output_layer).arrange(RIGHT, buff=1.5)
        all_layers.next_to(input_image, RIGHT, buff=1.2)

        # --- Add layers
        #self.add(input_image, all_layers)

        # Connect layers
        edges1 = self.connect_layers(input_neurons, hidden1)
        edges2 = self.connect_layers(hidden1, hidden2)
        edges3 = self.connect_layers(hidden2, output_layer)
        #self.add(*edges1)
        #self.add(*edges2)
        #self.add(*edges3)
        #self.play(*[Create(e) for e in edges], run_time=2)


        bars = self.create_output_bars(pred_output, output_layer)
        loss_terms = self.show_loss_terms(losses, pred_output, label, output_layer)
        # Total loss
        total_loss_text = Text(f"Total Loss: {total_loss:.3f}", font_size=20).next_to(loss_terms, UP, buff=0.3)

        all_content = Group(input_image, input_layer, hidden1, hidden2, output_layer, *edges1, *edges2, *edges3, bars, loss_terms, total_loss_text)
        #all_content.shift(LEFT * 1.5)
        all_content.move_to(ORIGIN)

        # Initially hide elements
        bars.set_opacity(0)
        loss_terms.set_opacity(0)
        total_loss_text.set_opacity(0)


        self.add(all_content)


        # Simulate forward pass activation coloring
        self.activate_layer(input_layer, delay=0.05)
        self.activate_layer(hidden1, delay=0.05)
        self.activate_layer(hidden2, delay=0.05)
        self.activate_layer(output_layer, delay=0.05)

        # Output bar chart + loss computation
        #self.play(*[GrowFromEdge(b, edge=DOWN) for b in bars], run_time=1)
        #self.play(FadeIn(bars), run_time=1)
        self.play(bars.animate.set_opacity(1), run_time=1.5)

        #self.play(*[Write(t) for t in loss_terms], run_time=1.5)
        #self.play(FadeIn(loss_terms), run_time=1.5)
        #self.play(FadeIn(total_loss_text))
        self.play(loss_terms.animate.set_opacity(1), run_time=1.5)
        self.play(total_loss_text.animate.set_opacity(1), run_time=1.5)

        # Simulate weight updates (flash some edges)
        self.simulate_weight_update(edges3)
        self.simulate_weight_update(edges2)
        self.simulate_weight_update(edges1)

        self.wait(2)

    def create_layer(self, n, spacing=0.4):
        """Creates a full layer with n neurons"""
        layer = VGroup()
        for i in range(n):
            y = (n / 2 - i) * spacing
            neuron = Circle(radius=0.1, color=WHITE, fill_opacity=0.3)
            neuron.move_to(np.array([0, y, 0]))
            layer.add(neuron)
        return layer

    def create_input_layer(self, total, visible, spacing):
        """Creates input layer showing only the first and last N visible neurons out of total neurons."""
        layer = VGroup()

        # Top neurons
        top_neurons = VGroup()
        for i in range(visible):
            neuron = Circle(radius=0.1, color=WHITE, fill_opacity=0.3)
            top_neurons.add(neuron)
        top_neurons.arrange(DOWN, buff=spacing)

        # Bottom neurons
        bottom_neurons = VGroup()
        for i in range(visible):
            neuron = Circle(radius=0.1, color=WHITE, fill_opacity=0.3)
            bottom_neurons.add(neuron)
        bottom_neurons.arrange(DOWN, buff=spacing)

        # Dots (in between)
        dots = Tex(r"\vdots", color=WHITE).scale(1.2)

        # Stack all vertically with custom gap
        layer_group = VGroup(top_neurons, dots, bottom_neurons).arrange(DOWN, buff=2*spacing)

        # Return full layer and neuron list
        all_neurons = list(top_neurons) + list(bottom_neurons)
        return layer_group, all_neurons

        # Optionally add a text label showing total neurons
        #label = Text(f"{total} neurons", font_size=18).next_to(layer, DOWN, buff=0.2)
        #layer.add(label)

    def connect_layers(self, layer1, layer2):
        edges = []
        for n1 in layer1:
            if isinstance(n1, Tex): continue  # skip ellipsis
            for n2 in layer2:
                e = Line(n1.get_center(), n2.get_center(), stroke_width=1.5, color=GRAY)
                edges.append(e)
        return edges

    def activate_layer(self, layer, delay=0.1):
        anims = []
        for node in layer:
            if isinstance(node, Tex): continue
            anims.append(node.animate.set_fill(RED, opacity=1))
        self.play(*anims, run_time=delay * len(anims))

    def create_output_bars(self, probs, ref_layer):
        bars = VGroup()
        max_p = max(probs)
        for i, p in enumerate(probs):
            bar = Rectangle(
                height=p * 2,
                width=0.3,
                fill_color=ORANGE if p == max_p else GRAY,
                fill_opacity=0.8,
                stroke_color=WHITE
            )
            bar.next_to(ref_layer[i], RIGHT, buff=0.3, aligned_edge=DOWN)
            bars.add(bar)

        return bars

    def show_loss_terms(self, losses, probs, label, ref_layer):
        terms = VGroup()
        for i, (l, p) in enumerate(zip(losses, probs)):
            color = RED if i == label else GRAY
            text = Text(f"-{int(i == label)}·log({p:.2f}) = {l:.2f}", font_size=16, color=color)
            text.next_to(ref_layer[i], RIGHT, buff=1.0)
            terms.add(text)
        return terms

    def simulate_weight_update(self, edges, flash_fraction=0.1):
        # Pick some random edges to flash
        n = int(len(edges) * flash_fraction)
        flash_edges = random.sample(edges, n)
        anims = []
        for e in flash_edges:
            anims.append(e.animate.set_color(YELLOW).set_stroke(width=2))
        self.play(*anims, run_time=2)
