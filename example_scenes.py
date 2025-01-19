from manim import *


class CreateCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
        self.play(Create(circle))  # show the circle on screen
        self.wait()


class AnimatedSquareToCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        square = Square()  # create a square

        self.play(Create(square))  # show the square on screen
        self.play(square.animate.rotate(PI / 4))  # rotate the square
        self.play(Transform(square, circle))  # transform the square into a circle
        self.play(
            square.animate.set_fill(PINK, opacity=0.5)
        )  # color the circle on screen


class MarkupTest(Scene):
    def construct(self):
        text = MarkupText(
            f'<span underline="double" underline_color="green">double green underline</span> in red text<span fgcolor="{YELLOW}"> except this</span>',
            color=RED,
            font_size=34
        )
        self.add(text)


class CorrectLaTeXSubstringColoring(Scene):
    def construct(self):
        equation = MathTex(
            r"e^x = x^0 + x^1 + \frac{1}{2} x^2 + \frac{1}{6} x^3 + \cdots + \frac{1}{n!} x^n + \cdots",
            substrings_to_isolate="x"
        )
        equation.set_color_by_tex("x", YELLOW)
        self.add(equation)


class MathTeXDemo(Scene):
    def construct(self):
        rtarrow0 = MathTex(r"\xrightarrow{x^6y^8}", font_size=96)
        rtarrow1 = Tex(r"$\xrightarrow{x^6y^8}$", font_size=96)

        self.add(VGroup(rtarrow0, rtarrow1).arrange(DOWN))


class PlotExample(Scene):
    def construct(self):
        # construct the axes
        ax_1 = Axes(
            x_range=[0.001, 6],
            y_range=[-8, 2],
            x_length=5,
            y_length=3,
            tips=False,
        )
        ax_2 = ax_1.copy()
        ax_3 = ax_1.copy()

        # position the axes
        ax_1.to_corner(UL)
        ax_2.to_corner(UR)
        ax_3.to_edge(DOWN)
        axes = VGroup(ax_1, ax_2, ax_3)

        # create the logarithmic curves
        def log_func(x):
            return np.log(x)

        # a curve without adjustments; poor interpolation.
        curve_1 = ax_1.plot(log_func, color=PURE_RED)

        # disabling interpolation makes the graph look choppy as not enough
        # inputs are available
        curve_2 = ax_2.plot(log_func, use_smoothing=False, color=ORANGE)

        # taking more inputs of the curve by specifying a step for the
        # x_range yields expected results, but increases rendering time.
        curve_3 = ax_3.plot(
            log_func, x_range=(0.001, 6, 0.001), color=PURE_GREEN
        )

        curves = VGroup(curve_1, curve_2, curve_3)

        self.add(axes, curves)
