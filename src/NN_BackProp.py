from manim import *
import numpy as np
import random

class BackpropagationScene(Scene):
    """
    Visualize one backpropagation step:
    Highlight edges layer by layer (output -> hidden2 -> hidden1 -> input).
    """

    def construct(self):
        self.camera.background_color = "#43506E"
        # --- Architecture
        input_visible = 6
        hidden1_size = 16
        hidden2_size = 16
        output_size  = 10

        # --- Build layers
        input_layer, input_neurons = self.create_input_layer(total=784, visible=input_visible, spacing=0.2)
        hidden1      = self.create_layer(hidden1_size, spacing=0.35)
        hidden2      = self.create_layer(hidden2_size, spacing=0.35)
        output_layer = self.create_layer(output_size,  spacing=0.45)

        net = VGroup(input_layer, hidden1, hidden2, output_layer).arrange(RIGHT, buff=3)
        net.scale_to_fit_width(config.frame_width * 0.9)
        net.scale_to_fit_height(config.frame_height * 0.6)
        net.move_to(ORIGIN)

        # --- Connect layers
        edges1 = self.connect_layers(input_neurons, hidden1, base_width=1.0)
        edges2 = self.connect_layers(hidden1, hidden2, base_width=1.0)
        edges3 = self.connect_layers(hidden2, output_layer, base_width=1.0)
        all_edges = VGroup(*edges1, *edges2, *edges3)

        # --- Title
        title = Text("Backpropagation", font_size=36, font="DejaVu Sans")
        title.to_edge(UP)

        # --- Braces for input/output layers
        input_brace = Brace(input_layer, LEFT, buff=0.2)
        input_label = Text("Input Layer", font_size=24, font="DejaVu Sans").next_to(
            input_brace, LEFT).rotate(PI/2)

        output_brace = Brace(output_layer, RIGHT, buff=0.2)
        output_label = Text("Output Layer", font_size=24, font="DejaVu Sans").next_to(
            output_brace, RIGHT).rotate(-PI/2)

        # --- Add all content
        self.add(title, input_layer, hidden1, hidden2, output_layer, all_edges,
                 input_brace, input_label, output_brace, output_label)

        # --- Animate backprop: highlight edges backwards
        self.backprop_step(output_layer, hidden2, edges3)
        self.backprop_step(hidden2, hidden1, edges2)
        self.backprop_step(hidden1, input_layer, edges1)

        self.wait(1)

    def backprop_step(self, layer_from, layer_to, edges, color="#ecfcca"):
        """
        Highlight a subset of edges in custom color to visualize gradient flow
        and show a large backward arrow beneath the network.
        """
        # choose 20% of edges randomly
        num_highlight = max(1, int(len(edges) * 0.2))
        highlight_edges = random.sample(edges, num_highlight)

        # highlight edges
        anims = [e.animate.set_color(color).set_stroke(width=2) for e in highlight_edges]
        self.play(LaggedStart(*anims, lag_ratio=0.005, run_time=1))

        # big backprop arrow drawn below the net (from right to left)
        arrow = Arrow(
            start=layer_from.get_center() + DOWN * 3,
            end=layer_to.get_center() + DOWN * 3,
            buff=0.4,
            stroke_width=6,
            color=color,
            max_tip_length_to_length_ratio=0.1
        )
        self.play(GrowArrow(arrow))
        self.wait(0.5)
        self.play(FadeOut(arrow))

    # ----------------- helpers -----------------
    def create_layer(self, n: int, spacing: float = 0.4):
        layer = VGroup()
        y0 = (n - 1) * spacing / 2
        for i in range(n):
            neuron = Circle(radius=0.10, color=WHITE, fill_opacity=0.25, stroke_width=1.2)
            neuron.move_to(np.array([0.0, y0 - i * spacing, 0.0]))
            layer.add(neuron)
        return layer

    def create_input_layer(self, total, visible, spacing):
        top = VGroup(*[Circle(radius=0.1, color=WHITE, fill_opacity=0.25, stroke_width=1.2) for _ in range(visible)]).arrange(DOWN, buff=spacing)
        bottom = VGroup(*[Circle(radius=0.1, color=WHITE, fill_opacity=0.25, stroke_width=1.2) for _ in range(visible)]).arrange(DOWN, buff=spacing)
        dots = Tex(r"\vdots", color=WHITE).scale(1.2)
        group = VGroup(top, dots, bottom).arrange(DOWN, buff=1.5*spacing)
        return group, list(top) + list(bottom)

    def connect_layers(self, layer1: VGroup, layer2: VGroup, base_width: float = 1.2, color=GRAY):
        edges = []
        for n1 in layer1:
            for n2 in layer2:
                e = Line(n1.get_center(), n2.get_center(), stroke_width=base_width, color=color)
                edges.append(e)
        return edges