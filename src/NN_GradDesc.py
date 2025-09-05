from manim import *
import numpy as np
import random

class GradientDescentEdgesScene(Scene):
    """
    Isolated neural network visualization that simulates three gradient descent steps
    by visibly changing edge (connection) thicknesses.
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
        net.scale_to_fit_width(config.frame_width * 1)
        net.scale_to_fit_width(config.frame_height * 1)
        net.move_to(ORIGIN)


        # --- Connect layers
        edges1 = self.connect_layers(input_neurons, hidden1, base_width=1.6)
        edges2 = self.connect_layers(hidden1,     hidden2, base_width=1.4)
        edges3 = self.connect_layers(hidden2,     output_layer, base_width=1.2)

        all_edges = VGroup(*edges1, *edges2, *edges3)

        # --- Title / step label
        title = Text("Gradientenabstieg eines Neuronalen Netzes", font_size=36, font="DejaVu Sans")
        title.to_edge(UP)
        step_label = Text("1. Durchlauf", font_size=28, font="DejaVu Sans").next_to(title, DOWN, buff=0.25)

        # --- Braces for input/output layers
        input_brace = Brace(input_layer, LEFT, buff=0.2)
        input_label = Text("Input Layer", font_size=24, font="DejaVu Sans").next_to(
            input_brace, LEFT).rotate(PI/2)

        output_brace = Brace(output_layer, RIGHT, buff=0.2)
        output_label = Text("Output Layer", font_size=24, font="DejaVu Sans").next_to(
            output_brace, RIGHT).rotate(-PI/2)

        # --- Show setup
        self.add(title, step_label, input_layer, hidden1, hidden2, output_layer, all_edges,
                 input_brace, input_label, output_brace, output_label)

        # --- Initialize pseudo-weights mapped to stroke widths
        rng = np.random.default_rng(42)
        widths = {e: e.get_stroke_width() + float(rng.normal(0, 0.25)) for e in all_edges}
        for e, w in widths.items():
            e.set_stroke(width=max(0.8, min(4.0, w)))

        # --- Helper to perform one "GD step"
        def gradient_step(step_idx: int):
            scale = 1.5 * (0.9 ** step_idx)  # stronger changes
            new_widths = {}

            strengthen_fraction = 0.18  # more edges get strengthened
            num_strengthen = max(1, int(len(all_edges) * strengthen_fraction))
            strengthen_edges = set(random.sample(list(all_edges), num_strengthen))

            anims = []
            for e in all_edges:
                delta = float(rng.normal(0.0, scale))
                if e in strengthen_edges:
                    delta += abs(float(rng.normal(0.15, 0.05)))
                target = widths[e] + delta
                target = max(1, min(8.0, target))
                new_widths[e] = target

                anims.append(e.animate.set_stroke(width=target))

            new_label = Text(f"{step_idx}. Durchlauf", font_size=28, font="DejaVu Sans").next_to(
                title, DOWN, buff=0.25)
            self.play(Transform(step_label, new_label), run_time=0.6)

            self.play(LaggedStart(*anims, lag_ratio=0.0015, run_time=1.5, rate_func=smooth))
            self.wait(0.3)

            widths.update(new_widths)

        # --- Perform three steps
        for s in range(1, 4):
            gradient_step(s)

        self.wait(1)

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
