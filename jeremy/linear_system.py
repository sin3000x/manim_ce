from manim import *
from manim.scene.vector_space_scene import X_COLOR, Y_COLOR


class LTDemo(LinearTransformationScene):
    def __init__(self, **kwargs):
        super().__init__(
            leave_ghost_vectors=True,
            **kwargs
        )
        self.A = [[2, 2], [1, 3]]
        self.matrix = Matrix(self.A, h_buff=1)
        self.matrix.get_columns()[0].set_color(X_COLOR)
        self.matrix.get_columns()[1].set_color(Y_COLOR)
        self.title = (
            VGroup(MathTex("A ="), self.matrix)
            .arrange()
            .to_edge(DOWN, buff=1)
            .add_background_rectangle()
        )

    def construct(self):
        self.apply_matrix(self.A)
        self.play(Write(self.title))
        self.wait()

class ChangeBasis(LinearTransformationScene):
    def __init__(self, **kwargs):
        super().__init__(
            leave_ghost_vectors=False,
            **kwargs
        )
        self.A = [[2, 2], [1, 3]]
        self.b = (-4, -1)
        self.x = np.linalg.solve(self.A, self.b)
        self.title = MathTex(
            r"A {{\vec{x}}} = {{\vec{b}}}",
            tex_to_color_map={'x': PINK, 'b': YELLOW}
        ).add_background_rectangle().to_edge(UP, buff=1)

        self.b_vec = Vector(self.b, color=YELLOW)
        self.b_vec.label = (
            MathTex(r"\vec{b}", color=YELLOW)
            .add_background_rectangle()
            .next_to(self.b_vec.get_end(), DL)
        )

        self.x_vec = Vector(self.x, color=PINK)
        self.x_vec.label = (
            MathTex(r"\vec{x}", color=PINK)
            .add_background_rectangle()
            .next_to(self.x_vec.get_end(), LEFT)
        )
        self.add_moving_mobject(self.x_vec)

    def construct(self):
        self.play(Write(self.title))
        self.play(GrowArrow(self.b_vec))
        self.play(Write(self.b_vec.label))

        self.play(GrowArrow(self.x_vec))
        self.play(Write(self.x_vec.label))
        self.apply_matrix(self.A)
        self.wait()

