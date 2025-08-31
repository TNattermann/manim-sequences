from manim import *
import numpy as np
import random

class FullTrainingCycleScene(Scene):
    """
    Drop‑in full training cycle with a centered network (no left/right extras)
    and an adaptive description strip *below* the network.

    Stages:
      1) Forward pass  – neurons activate and a subset of forward edges highlight
      2) Loss          – show descriptive text (or image/video) in the strip
      3) Backprop      – highlight edges backwards + big arrow *below* the net
      4) Gradient step – visibly change edge thickness/color to simulate updates

    How to add media to the description strip:
      - Replace the Text(...) objects in show_desc(...) calls with ImageMobject(...) or
        VideoMobject(...) and the positioning will still work.
    """

    # === Tweaks you may want to customize ===
    FORWARD_EDGE_COLOR = "#d6bcfa"            # flow hint during forward pass
    BACKPROP_COLOR_HEX = "#ecfcca"       # your custom hex color for backprop highlights/arrow
    GD_MIN_W, GD_MAX_W = 1.0, 6.0        # stroke width bounds during gradient step

    def construct(self):
        # ---------------- Network ----------------
        self.camera.background_color = "#43506E"
        input_visible = 5
        hidden1_size  = 16
        hidden2_size  = 16
        output_size   = 10

        # Layers (input shown as top/bottom groups with ellipsis)
        input_layer, input_neurons = self.create_input_layer(total=784, visible=input_visible, spacing=0.3)
        hidden1      = self.create_layer(hidden1_size)
        hidden2      = self.create_layer(hidden2_size)
        output_layer = self.create_layer(output_size, spacing=0.6)

        # Arrange only the layers first
        net_layers = VGroup(input_layer, hidden1, hidden2, output_layer).arrange(RIGHT, buff=3.5)

        # Connect AFTER arranging, then group everything so scaling considers edges too
        edges1 = self.connect_layers(input_neurons, hidden1)
        edges2 = self.connect_layers(hidden1, hidden2)
        edges3 = self.connect_layers(hidden2, output_layer)
        full_net = VGroup(net_layers, *edges1, *edges2, *edges3)

        # Safe downscale‑only fit to leave room for the description strip below
        MAX_W = config.frame_width  * 0.92
        MAX_H = config.frame_height * 0.6
        scale_w = MAX_W / full_net.width
        scale_h = MAX_H / full_net.height
        scale   = min(scale_w, scale_h, 1.0)  # never upscale
        full_net.scale(scale)
        full_net.to_edge(UP, buff=0.6)

        self.add(full_net)

        # Baseline below the network for arrows and the description strip
        baseline_y = full_net.get_bottom()[1] - 1.2

        # A container for the changing description (text/image/video)
        self.desc_group = Group()
        self.add(self.desc_group)

        # --------------- Stage 1: Forward pass ---------------
        forward_text = MarkupText("Vorhersage der KI\n"
                            "          =\n"
                            "Bogenschuss",
                            font_size=28)
        forward_text.align_on_border(UP)
        forward_media = ImageMobject("media_input/FP_transparent.png").scale(0.4)
        self.show_desc(forward_text, forward_media, baseline_y)
        self.activate_layer(input_layer, delay=0, edge_list=edges1, color=self.FORWARD_EDGE_COLOR)
        self.activate_layer(hidden1,     delay=0, edge_list=edges2, color=self.FORWARD_EDGE_COLOR)
        self.activate_layer(hidden2,     delay=0, edge_list=edges3, color=self.FORWARD_EDGE_COLOR)
        self.activate_layer(output_layer, delay=0)

        self.wait(3)
        self.reset_net_styles([input_layer, hidden1, hidden2, output_layer], edges1 + edges2 + edges3)

        # --------------- Stage 2: Loss ---------------
        # (Replace Text with ImageMobject("path.png") or VideoMobject("clip.mp4") if desired)
        loss_text = MarkupText("Abgleich Vorhersage - Label\n"
                               "          =\n"
                               "Distanz zur Zielscheibe",
                               font_size=28)
        loss_text.align_on_border(UP)
        loss_media = ImageMobject("media_input/Loss_transparent.png").scale(0.35)
        self.show_desc(loss_text, loss_media, baseline_y)
        self.wait(4)

        # --------------- Stage 3: Backpropagation ---------------
        backprop_text = MarkupText("Ermittlung der Gewichtsanpassungen\n"
                             "          =\n"
                             "Trainerfeedback",
                             font_size=28)
        backprop_text.align_on_border(UP)
        backprop_media = ImageMobject("media_input/Backprop_transparent.png").scale(0.4)
        self.show_desc(backprop_text, backprop_media, baseline_y)
        self.backprop_step(output_layer, hidden2, edges3, baseline_y, color=self.BACKPROP_COLOR_HEX)
        self.backprop_step(hidden2,     hidden1, edges2, baseline_y, color=self.BACKPROP_COLOR_HEX)
        self.backprop_step(hidden1,     input_layer, edges1, baseline_y, color=self.BACKPROP_COLOR_HEX)
        self.wait(3)
        self.reset_net_styles([input_layer, hidden1, hidden2, output_layer], edges1 + edges2 + edges3)

        # --------------- Stage 4: Gradient descent update ---------------
        graddesc_text = MarkupText("Durchführung der Gewichtsanpassungen\n"
                                   "          =\n"
                                   "Technik korrigieren",
                                   font_size=28)
        graddesc_text.align_on_border(UP)
        graddesc_media = ImageMobject("media_input/GradDesc_transparent.png").scale(1.0)
        self.show_desc(graddesc_text, graddesc_media, baseline_y)
        all_edges = edges1 + edges2 + edges3
        self.gradient_descent_update(all_edges)

        self.wait(3)

    # ---------------- Helpers ----------------
    def show_desc(
            self,
            text_mobj: Mobject,
            media_mobj: Mobject,
            baseline_y: float,
            *,
            image_scale: float | None = None,
            buff: float = 1.0,
            animate: bool = True,
    ):
        """
        Minimal: text (left) + image (right), placed below the net at baseline_y.
        - No auto-scaling, no overflow handling (you control sizes).
        - Optional quick animation toggle.
        """

        # Optional: scale the image explicitly here (or do it before calling)
        if image_scale is not None:
            media_mobj.scale(image_scale)

        group = Group(text_mobj, media_mobj)
        group.arrange(RIGHT, buff=buff)
        group.move_to(np.array([0, baseline_y, 0]))

        # Swap current description (fast, minimal)
        if not hasattr(self, "current_desc") or self.current_desc is None:
            self.current_desc = group
            self.add(group) if not animate else self.play(FadeIn(group), run_time=0.2)
        else:
            old = self.current_desc
            if animate:
                self.play(FadeOut(old), run_time=0.15)
                self.remove(old)
                self.current_desc = group
                self.play(FadeIn(group), run_time=0.2)
            else:
                self.remove(old)
                self.current_desc = group
                self.add(group)

    def backprop_arrow(self, layer_from: VGroup, layer_to: VGroup, baseline_y: float, color=YELLOW):
        start = np.array([layer_from.get_center()[0], baseline_y, 0])
        end   = np.array([layer_to.get_center()[0],   baseline_y, 0])
        return Arrow(start, end, buff=0.2, stroke_width=6, color=color, max_tip_length_to_length_ratio=0.1)

    def backprop_step(self, layer_from, layer_to, edges, baseline_y: float, color="#FF5733"):
        # Highlight a subset of edges (backward step)
        num = max(1, int(len(edges) * 0.2))
        chosen = random.sample(edges, num)
        anims = [e.animate.set_color(color).set_stroke(width=3.5) for e in chosen]
        self.play(LaggedStart(*anims, lag_ratio=0.003, run_time=0.5))

        # Big arrow below the network pointing backward (right → left)
        arrow = self.backprop_arrow(layer_from, layer_to, baseline_y+1, color=color)
        self.play(GrowArrow(arrow), run_time=0.5)
        self.wait(0.2)
        self.play(FadeOut(arrow), run_time=0.4)

    def gradient_descent_update(self, edges):
        rng = np.random.default_rng(42)
        widths = {e: e.get_stroke_width() for e in edges}
        anims = []
        for e in edges:
            delta = float(rng.normal(0.0, 2))
            target = max(self.GD_MIN_W, min(self.GD_MAX_W, widths[e] + delta))
            # 👇 Only change stroke width, keep color fixed
            anims.append(e.animate.set_stroke(width=target, color=GRAY))
        self.play(LaggedStart(*anims, lag_ratio=0.001, run_time=2.2, rate_func=smooth))

    def create_layer(self, n, spacing=0.4):
        layer = VGroup()
        for i in range(n):
            y = (n / 2 - i) * spacing
            neuron = Circle(radius=0.1, color=WHITE, fill_opacity=0.3)
            neuron.move_to(np.array([0, y, 0]))
            layer.add(neuron)
        return layer

    def create_input_layer(self, total, visible, spacing):
        top = VGroup(*[Circle(radius=0.1, color=WHITE, fill_opacity=0.3) for _ in range(visible)]).arrange(DOWN, buff=spacing)
        bottom = VGroup(*[Circle(radius=0.1, color=WHITE, fill_opacity=0.3) for _ in range(visible)]).arrange(DOWN, buff=spacing)
        dots = Tex(r"\vdots", color=WHITE).scale(1.2)
        group = VGroup(top, dots, bottom).arrange(DOWN, buff=2*spacing)
        return group, list(top) + list(bottom)

    def connect_layers(self, layer1, layer2):
        edges = []
        for n1 in layer1:
            if isinstance(n1, Tex):
                continue
            for n2 in layer2:
                e = Line(n1.get_center(), n2.get_center(), stroke_width=1.5, color=GRAY)
                edges.append(e)
        return edges

    def activate_layer(self, layer, delay=0.1, edge_list=None, color=BLUE):
        anims = []
        for node in layer:
            if isinstance(node, Tex):
                continue
            anims.append(node.animate.set_fill(color, opacity=1))
        if edge_list:
            k = max(1, len(edge_list) // 5)
            anims += [e.animate.set_color(color).set_stroke(width=2.5) for e in random.sample(edge_list, k)]
        self.play(*anims, run_time=0.25)


    def reset_net_styles(self, layers, edges):
        for e in edges:
            e.set_color(GRAY).set_stroke(width=1.5)
        for layer in layers:
            for node in layer:
                if isinstance(node, Tex):
                    continue
                node.set_fill(WHITE, opacity=0.3)

