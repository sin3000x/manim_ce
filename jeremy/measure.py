import random
from functools import cache

from manim import *

import numpy as np


def find_intervals_by_range(f, x_min, x_max, y_min, y_max, epsilon=0.001):
    # Create a range of x values with step size epsilon using numpy
    x_values = np.arange(x_min, x_max, epsilon)
    y_values = f(x_values)

    mask: np.ndarray = (y_min <= y_values) & (y_values <= y_max)

    # Find the indices where the mask is True (i.e., where the condition is satisfied)
    # We also need to identify the start and end points of each subinterval.
    diff = np.diff(mask.astype(int))  # Convert True/False to 1/0 and compute the difference
    start_indices = np.where(diff == 1)[0] + 1  # Start of new intervals
    end_indices = np.where(diff == -1)[0]  # End of intervals

    # If the range starts with a valid subinterval, add it as well
    if mask[0]:
        start_indices = np.insert(start_indices, 0, 0)

    # If the range ends with a valid subinterval, include xmax in the last interval
    if mask[-1]:
        end_indices = np.append(end_indices, len(x_values) - 1)

    # Pair the start and end indices to get the subintervals
    intervals = [(x_values[start], x_values[end]) for start, end in zip(start_indices, end_indices)]

    return intervals


def x_generator(intervals):
    for interval in intervals:
        yield from interval


class LebesgueIntegral(Scene):
    def __init__(self):
        super().__init__()
        self.axes = Axes(tips=False, x_range=(-4, 4), y_range=(-2, 2))
        self.x_min, self.x_max = self.axes.x_range[0], self.axes.x_range[1]
        self.nice_graph = self.axes.plot(self.nice_function, use_vectorized=True, color=RED)
        self.bad_graph = self.axes.plot(
            self.bad_function, use_vectorized=True, x_range=(self.x_min, self.x_max, 1e-3), color=BLUE
        )

    def get_nice_intervals(self, y_min, y_max):
        return find_intervals_by_range(f=self.nice_function, x_min=self.x_min, x_max=self.x_max, y_min=y_min, y_max=y_max)

    def get_bad_intervals(self, y_min, y_max):
        return find_intervals_by_range(
            f=self.bad_function,
            x_min=self.x_min, x_max=self.x_max,
            y_min=y_min, y_max=y_max,
            epsilon=1e-6
        )

    @staticmethod
    def nice_function(arr):
        return arr ** 2  / 8

    @staticmethod
    def bad_function(arr):
        mask = np.abs(arr) > 0.08
        result = np.where(mask, 2 * np.sin(50 / arr), 0)
        return result

    def get_vertical_lines(self, graph, intervals, **kwargs) -> VGroup:
        graph_points = (self.axes.input_to_graph_point(x, graph) for x in x_generator(intervals))
        vertical_lines = list(self.axes.get_vertical_line(graph_point, **kwargs).rotate(PI) for graph_point in graph_points)
        return VGroup(*vertical_lines)

    def get_subinterval_lines(self, subintervals, **kwargs) -> VGroup:
        def get_line(x1, x2):
            return Line(self.axes.c2p(x1, 0), self.axes.c2p(x2, 0), **kwargs)

        return VGroup(
            *[get_line(x1, x2) for (x1, x2) in subintervals]
        )

    def get_nice_lebesgue_rectangles(self, intervals):
        rectangles = VGroup()
        for (x1, x2) in intervals:
            if x1 < 0 < x2:
                sample_input = 0
            else:
                sample_input = x1 if self.nice_function(x1) <= self.nice_function(x2) else x2
            graph_point = self.axes.input_to_graph_point(sample_input, self.nice_graph)
            points = VGroup(
                *list(
                    map(
                        VectorizedPoint,
                        [
                            self.axes.c2p(x1, 0),
                            self.axes.c2p(x2, 0),
                            graph_point,
                        ],
                    ),
                )
            )

            rect = Rectangle().replace(points, stretch=True)
            rectangles.add(rect)
        return rectangles

    def construct_nice_function(self):
        self.add(self.axes, self.nice_graph)
        self.wait()
        riemann_rectangles = self.axes.get_riemann_rectangles(
            graph=self.nice_graph, dx=0.5,
        )
        self.play(
            LaggedStart(
                *[FadeIn(rectangle) for rectangle in riemann_rectangles]
            )
        )
        self.wait()

        example_interval = self.axes.plot(lambda x: 0, x_range=(1, 1.5, 0.5))
        brace = Brace(example_interval, DOWN)
        self.play(GrowFromCenter(brace))
        assumption = Tex(
            r"\kaishu 足够小时，\\需要{{$f(x)$}}的振幅很小",
            tex_template = TexTemplateLibrary.ctex,
        ).next_to(brace, DOWN).set_color_by_tex_to_color_map(
            {'f': RED}
        ).add_background_rectangle(opacity=1)
        self.play(Write(assumption))
        self.wait()

        self.play(
            FadeOut(
                VGroup(riemann_rectangles, brace, assumption)
            )
        )
        self.wait()

        y_intervals = ((1, 1.5), (0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1), (1, 1.2), (1.2, 1.4))
        horizontal_lines = {
            interval: VGroup(*[
                DashedLine(
                    self.axes.c2p(self.x_min, y), self.axes.c2p(self.x_max, y),
                    dashed_ratio=0.5, #dash_length=0.2
                )
                for y in interval
            ])
            for interval in y_intervals
        }
        x_intervals_dict = {
            y_interval: self.get_nice_intervals(*y_interval) for y_interval in y_intervals
        }
        interval_lines = {
            y_interval: self.get_subinterval_lines(x_intervals, color=YELLOW, stroke_width=8)
            for y_interval, x_intervals in x_intervals_dict.items()
        }
        vertical_lines = {
            y_interval: self.get_vertical_lines(graph=self.nice_graph, intervals=x_intervals, color=RED)
            for y_interval, x_intervals in x_intervals_dict.items()
        }
        rectangles_dict = {
            y_interval: self.get_nice_lebesgue_rectangles(x_intervals_dict[y_interval])
            for y_interval in y_intervals
        }
        colors = color_gradient((GREEN, BLUE), len(y_intervals))
        colors[0] = BLUE
        for color, y_interval in zip(colors, y_intervals):
            for rectangle in rectangles_dict[y_interval]:
                rectangle.set_style(
                    fill_color=color,
                    stroke_color=BLACK,
                    fill_opacity=.9,
                    stroke_opacity=0,
                )

        # start
        initial_interval = y_intervals[0]
        initial_horizontal_lines = horizontal_lines[initial_interval]
        initial_interval_lines = interval_lines[initial_interval]
        horizontal_labels = [MathTex("y_1"), MathTex("y_2")]
        for label, horizontal_line in zip(horizontal_labels, initial_horizontal_lines):
            label.set_color(horizontal_line.get_color())
            label.next_to(horizontal_line, LEFT)
        interval_labels = [MathTex("E_1"), MathTex("E_2")]
        for label, interval in zip(interval_labels, initial_interval_lines):
            label.set_color(interval.get_color())
            label.next_to(interval, DOWN)
        self.play(*[Create(line) for line in initial_horizontal_lines])
        self.play(*[Write(label) for label in horizontal_labels])
        self.wait()

        self.play(*[Create(line) for line in vertical_lines[initial_interval]])
        self.play(
            FadeIn(initial_interval_lines),
            *[Write(label) for label in interval_labels]
        )
        self.wait()

        # explain for calc
        self.play(DrawBorderThenFill(rectangles_dict[initial_interval]))
        self.wait()

        area_tex = MathTex(
            r"S=y_1\cdot{{m(E_1\cup E_2)}}",
            substrings_to_isolate=['S'],
            font_size=60,
            tex_template=TexTemplateLibrary.ctex,
            stroke_width=2,
        ).move_to(DOWN).set_color_by_tex_to_color_map(
            {'S': BLUE, 'E': YELLOW}
        ).add_background_rectangle(opacity=1, buff=.3)
        self.play(Write(area_tex))
        self.wait()

        self.play(
            FadeOut(
                VGroup(
                    initial_interval_lines, initial_horizontal_lines, vertical_lines[initial_interval],
                    *horizontal_labels, *interval_labels,
                    rectangles_dict[initial_interval], area_tex
                )
            )
        )
        for y_interval in y_intervals[1:]:
            self.play(*[Create(line) for line in horizontal_lines[y_interval]])
            self.play(*[Create(line) for line in vertical_lines[y_interval]])
            self.play(
                *[Create(line) for line in interval_lines[y_interval]],
                FadeIn(rectangles_dict[y_interval]),
                horizontal_lines[y_interval].animate.set_stroke(opacity=0.1),
            )
        self.wait()

        self.play(
            *[FadeOut(mob) for mob in self.mobjects if mob not in (self.axes,)]
        )

    def construct_bad_function(self):
        y_min, y_max = 1, 1.5
        self.play(FadeIn(self.bad_graph))
        horizontal_lines = VGroup(
            *[
                DashedLine( self.axes.c2p(self.x_min, y_min), self.axes.c2p(self.x_max, y_min), ),
                DashedLine(self.axes.c2p(self.x_min, y_max), self.axes.c2p(self.x_max, y_max), )
            ]
        )
        intervals = self.get_bad_intervals(y_min, y_max)
        interval_lines = self.get_subinterval_lines(intervals, color=YELLOW, stroke_width=8)
        vertical_lines = self.get_vertical_lines(graph=self.bad_graph, intervals=intervals, color=RED)
        self.play(*[Create(line) for line in horizontal_lines])
        self.play(*[Create(line) for line in vertical_lines])
        self.play(FadeIn(interval_lines))
        self.wait()

        self.play( *[FadeOut(mob) for mob in self.mobjects] )

    def construct(self):
        self.construct_nice_function()
        self.construct_bad_function()
        question = MathTex(
            r"m({{[0,1]\cap\mathbb{Q}}})=\,{{?}}", font_size=140, stroke_width=2
        ).set_color_by_tex_to_color_map({'Q': YELLOW, '?': YELLOW})
        self.play(Write(question))
        self.wait()


class LengthMapping(Scene):
    def __init__(self):
        super().__init__()
        self.example_font_size = 80
        self.sets = VGroup(
            *[MathTex(t, font_size=self.example_font_size, stroke_width=4)
              for t in (r"\varnothing", r"[0,1]\cup[3,4]", r"[1, +\infty)")]
        ).arrange(
            DOWN, buff=1, aligned_edge=RIGHT
        ).set_color(YELLOW).shift(UP).set_x(-3)

        self.lengths = VGroup(
            *[MathTex(t, font_size=self.example_font_size, stroke_width=4)
              for t in ("0", "2", r"+\infty")]
        ).set_color(BLUE).arrange(
            DOWN, aligned_edge=LEFT
        ).set_x(3)
        for l, s in zip(self.lengths, self.sets):
            l.set_y(s.get_y())

        self.arrows = VGroup(
            *[Arrow(s.get_right(), l.get_left(), buff=0.5)
              for (s, l) in zip(self.sets, self.lengths)]
        )

        self.mapping = MathTex(
            r"m^*\colon {{2^\mathbb{R}}}\to{{[0,+\infty]}}",
            font_size=90,
            stroke_width=3
        ).to_edge(DOWN, buff=1).set_color_by_tex_to_color_map(
            {'2': YELLOW, 'infty': BLUE}
        )

    def construct(self):
        self.add(self.sets)
        self.wait()
        for arrow, length in zip(self.arrows, self.lengths):
            self.play(GrowArrow(arrow))
            self.play(Write(length))
        self.wait()
        self.play(Write(self.mapping))
        self.wait()
        self.play(Wiggle(self.mapping[1], scale_value=1.2, n_wiggles=8))
        self.play(Wiggle(self.mapping[1], scale_value=1.2, n_wiggles=8))
        self.wait()

class LeftParen(MathTex):
    def __init__(self, **kwargs):
        super().__init__("(", **kwargs)
        self.center()

    def get_center(self):
        return VMobject.get_center(self) + 0.04*LEFT

class RightParen(MathTex):
    def __init__(self, **kwargs):
        super().__init__(")", **kwargs)
        self.center()

    def get_center(self):
        return VMobject.get_center(self) + 0.04*RIGHT


class OpenInterval(VGroup):
    def __init__(self, center_point = ORIGIN, width: float = 2, **kwargs):
        left = LeftParen(**kwargs).shift(LEFT*width/2)
        right = RightParen(**kwargs).shift(RIGHT*width/2)
        super().__init__(left, right)
         # scale_factor = width / 2.0
        # self.stretch(scale_factor, 0)
        # self.stretch(0.5+0.5*scale_factor, 1)
        self.shift(center_point)

class OuterMeasureIntro(Scene):
    def __init__(self):
        super().__init__()
        self.interval_length = Tex(
            "对{{$I=(a,b)$}}，定义长度{{$|I|=b-a$}}.",
            tex_template=TexTemplateLibrary.ctex,
            font_size=60,
            stroke_width=2,
        ).to_edge(UP, buff=.5).set_color_by_tex_to_color_map({'I': BLUE})

        self.number_line = NumberLine()
        self.example_set = VGroup(
            Line(self.number_line.number_to_point(-4), self.number_line.number_to_point(-2), stroke_width=8),
            Line(self.number_line.number_to_point(0), self.number_line.number_to_point(1), stroke_width=8),
            Dot(self.number_line.number_to_point(2)),
            Dot(self.number_line.number_to_point(4)),
        ).set_color(YELLOW)
        self.interval_stroke_width = 3
        self.interval_font_size = 80
        self.interval_color = BLUE
        self.cover1 = VGroup(
            self.get_interval(-3, 2.5),
            self.get_interval(0, 2),
            self.get_interval(3, 2.5),
        )
        self.cover1_label = self.get_cover_labels(self.cover1)
        self.length1 = MathTex("l=|I_1|+|I_2|+|I_3|").to_edge(DOWN, buff=1)
        self.cover_def = MathTex( r"{{E}}\subseteq{{\bigcup I_i}}", font_size=60 ).next_to(self.interval_length, DOWN, buff=.8).set_color_by_tex_to_color_map( {'E': YELLOW, 'I': BLUE} )

        self.cover2 = VGroup( self.get_interval(0, 10), )
        self.cover2_label = self.get_cover_labels(self.cover2)
        self.length2 = MathTex("l=|I_1|").to_edge(DOWN, buff=1)

        self.epsilon = ValueTracker(0.5)
        self.cover3 = VGroup(
            self.get_interval(-3, 2),
            self.get_interval(0.5, 1),
            always_redraw(
                lambda: self.get_interval(2, self.epsilon.get_value())
            ),
            always_redraw(
                lambda: self.get_interval(4, self.epsilon.get_value())
            ),
        )
        self.cover3_label = always_redraw(lambda: self.get_cover_labels(self.cover3))
        self.length3 = MathTex("l=|I_1|+|I_2|+|I_3|+|I_4|").to_edge(DOWN, buff=1)

        self.lebesgue_def = MathTex(
            r"m^*(E)=\inf\left\{\sum |I_i|\colon E\subseteq \bigcup I_i\right\}",
            font_size=60
        ).to_edge(DOWN, buff=.6)
        for e_idx in (3, 16):
            self.lebesgue_def[0][e_idx].set_color(YELLOW)
        for i_idx in (slice(10, 15), slice(18, 21)):
            self.lebesgue_def[0][i_idx].set_color(BLUE)
        # self.add(index_labels(self.lebesgue_def[0]))

    @staticmethod
    def get_cover_labels(cover: VGroup):
        arrows = VGroup()
        interval: OpenInterval
        for i, interval in enumerate(cover, start=1):
            arrow = DoubleArrow(
                start=interval.get_left(), end=interval.get_right(), buff=0, stroke_width=2,
            ).set_y(-0.8)
            label = MathTex(fr"|I_{i}|").next_to(arrow, DOWN).set_y(-1.3)
            arrows.add(VGroup(arrow, label))
        return arrows

    def grow_intervals(self, cover, lag_ratio: float = 0.2):
        self.play(
            LaggedStart(
                *[GrowFromCenter(interval) for interval in cover],
                lag_ratio=lag_ratio
            )
        )

    def write_label(self, label):
        self.play(
            *[GrowFromCenter(group) for group in label]
        )

    def get_interval(self, center_point, width):
        return OpenInterval(
            center_point=self.number_line.number_to_point(center_point),
            width=width,
            stroke_width=self.interval_stroke_width,
            font_size=self.interval_font_size,
        ).set_color(self.interval_color)

    def construct(self):
        self.play(
            Write(self.interval_length),
            Create(self.number_line)
        )
        self.wait()

        # example set
        self.play(Create(self.example_set))
        self.wait()

        # first cover
        self.grow_intervals(self.cover1)
        self.wait()
        self.play(Write(self.cover_def))
        self.wait()
        self.write_label(self.cover1_label)
        self.wait()
        self.play(Write(self.length1))
        self.wait()

        # second cover
        self.play(
            ReplacementTransform(self.cover1, self.cover2),
            ReplacementTransform(self.cover1_label, self.cover2_label),
            ReplacementTransform(self.length1, self.length2),
        )
        self.wait()

        # third cover
        self.play(
            FadeTransform(self.cover2, self.cover3),
            FadeTransform(self.cover2_label, self.cover3_label),
            FadeTransform(self.length2, self.length3),
        )
        self.wait()
        self.play(
            FadeOut(self.length3),
            self.cover_def.copy().animate.move_to(self.lebesgue_def[0][16:21])
        )
        self.play(FadeIn(self.lebesgue_def[0][10:16]))
        self.wait()
        self.play(
            Write(self.lebesgue_def[0][6:10]),
            Write(self.lebesgue_def[0][21]),
        )
        arrows = VGroup(*[
            Arrow(ORIGIN, DOWN, stroke_color=YELLOW).next_to(self.example_set[i], UP, buff=.5)
            for i in (-1, -2)
        ])
        self.add(arrows)
        self.play( self.epsilon.animate.set_value(0.1), run_time=5 )
        self.wait()
        self.play(Write(self.lebesgue_def[0][:6]))
        self.wait()

class TwoDimension(Scene):
    def __init__(self):
        super().__init__()
        self.rectangle_area = Tex(
            r"对{{$I=(a,b)\times(c,d)$}}，定义面积{{$|I|=(b-a)(d-c)$}}.",
            tex_template=TexTemplateLibrary.ctex,
            font_size=50,
            stroke_width=2,
        ).to_edge(UP, buff=.5).set_color_by_tex_to_color_map({'I': BLUE})
        self.curve = VGroup().set_points_smoothly(
            [UP, LEFT*3, LEFT*3+DOWN*2, DOWN*3.5, RIGHT*2+DOWN*2, RIGHT*3+DOWN, UP]
        ).shift(UP).set_style(
            fill_color=YELLOW, stroke_color=YELLOW, fill_opacity=.2
        )
        self.cover = VGroup(
            # *[self.get_rectangle(
            #     random.random() * 5, random.random() * 5, ()
            # )]
            self.get_rectangle(4, 3.5, (-3, 0, 0)),
            self.get_rectangle(4, 2, (-0.5, 0.5, 0)),
            self.get_rectangle(3, 3, (1.7, 0.6, 0)),
            self.get_rectangle(2, 3, (1, -0.9, 0)),
            self.get_rectangle(2, 4, (0, -2.5, 0)),
        )

    def get_rectangle(self, height, width, center):
        return DashedVMobject(
            Rectangle(height=height, width=width, color=BLUE),
            dashed_ratio=0.7, num_dashes=30
        ).move_to(center)

    def construct(self):
        self.play(Write(self.rectangle_area))
        self.wait()
        self.play(DrawBorderThenFill(self.curve))
        self.wait()
        for rectangle in self.cover:
            self.add(rectangle)
            self.wait(.5)
        self.wait()


class OuterMeasureCalc(Scene):
    def __init__(self):
        super().__init__()
        self.title = Title("\\heiti Lebesgue外测度", tex_template=TexTemplateLibrary.ctex)
        self.lebesgue_def = MathTex(
            r"m^*(E)=\inf\left\{\sum |I_i|\colon E\subseteq \bigcup I_i\right\}",
            font_size=60, stroke_width=1
        ).next_to(self.title, DOWN, buff=.5)
        for e_idx in (3, 16):
            self.lebesgue_def[0][e_idx].set_color(YELLOW)
        for i_idx in (slice(10, 15), slice(18, 21)):
            self.lebesgue_def[0][i_idx].set_color(BLUE)
        self.number_line = NumberLine()

        self.single_dot = Dot(self.number_line.number_to_point(0)).set_color(YELLOW)
        self.single_cover = OpenInterval(
            center_point=self.single_dot.get_center(), width=.5,
            stroke_width=3, font_size=80
        ).set_color(BLUE)
        self.single_epsilon = MathTex("\\varepsilon", color=BLUE, font_size=70).next_to(self.single_cover, DOWN)
        self.zero_singleton = MathTex(r"m^*(\{x\})=0", font_size=70).next_to(self.single_epsilon, DOWN, buff=1)
        self.zero_singleton[0][3:6].set_color(YELLOW)

        self.n_dot = VGroup(*[Dot(self.number_line.number_to_point(i)) for i in (-3, 0, 4)]).set_color(YELLOW)
        self.n_cover = VGroup(*[OpenInterval(
            center_point=dot.get_center(), width=.5,
            stroke_width=3, font_size=80
        ).set_color(BLUE) for dot in self.n_dot])
        self.n_epsilon = VGroup(*[
            MathTex(r"\tfrac{\varepsilon}{3}", color=BLUE, font_size=70).next_to(cover, DOWN)
            for cover in self.n_cover
        ])
        self.zero_finite = MathTex(r"m^*(\{x_1,\ldots,x_n\})=0", font_size=70).move_to(self.zero_singleton)
        self.zero_finite[0][3:14].set_color(YELLOW)

        self.infinite_dot = VGroup(*[Dot(self.number_line.number_to_point(i)) for i in range(-7, 8)]).set_color(YELLOW)
        self.infinite_cover = VGroup(*[OpenInterval(
            center_point=dot.get_center(), width=.5,
            stroke_width=3, font_size=80
        ).set_color(BLUE) for dot in self.infinite_dot])
        self.infinite_epsilon = MathTex("\\varepsilon_n", color=BLUE, font_size=70).next_to(self.single_cover, DOWN)
        self.question = MathTex(r"m^*({{\mathbb{Z}}})=?", font_size=70).move_to(self.zero_singleton).set_color_by_tex_to_color_map({'Z': YELLOW})

    def construct(self):
        self.add(self.title)
        self.play(Write(self.lebesgue_def))
        self.wait()
        self.play(Create(self.number_line))

        # single
        self.play(GrowFromCenter(self.single_dot))
        self.wait()
        self.play(GrowFromCenter(self.single_cover))
        self.play(Write(self.single_epsilon))
        self.wait()
        self.play(Write(self.zero_singleton))
        self.wait()

        # n
        self.play(
            FadeOut(
                VGroup(self.single_dot, self.single_cover, self.single_epsilon, self.zero_singleton),
            ),
            FadeIn(
                VGroup(self.n_dot, self.n_cover, self.n_epsilon, self.zero_finite)
            ),
            run_time=2
        )
        self.wait()

        # infinite
        self.play(
            FadeOut(self.n_cover),
            FadeOut(self.n_epsilon),
            ReplacementTransform(self.n_dot, self.infinite_dot),
        )
        self.wait()
        self.play(FadeTransform(self.zero_finite, self.question))
        self.wait()
        self.play(FadeIn(self.infinite_cover), FadeIn(self.infinite_epsilon))
        self.wait()


class Infinity(Scene):
    def __init__(self):
        super().__init__()
        self.font_size = 60
        self.stroke_width = 2
        self.finite = Tex(
            r"有限集：$E=\{x_1,\ldots,x_n\}$",
            tex_template=TexTemplateLibrary.ctex, font_size=self.font_size,
            stroke_width=self.stroke_width
        )
        self.finite[0][4:].set_color(YELLOW)
        self.countable = Tex(
            r"可数集：$E=\{x_1,x_2,x_3,\cdots\}$",
            tex_template=TexTemplateLibrary.ctex, font_size=self.font_size,
            stroke_width = self.stroke_width
        )
        self.countable[0][4:].set_color(YELLOW)
        self.uncountable = Tex(
            r"不可数集：$(0,1),~\mathbb{R},~\mathbb{C},\cdots$",
            tex_template=TexTemplateLibrary.ctex, font_size=self.font_size,
            stroke_width=self.stroke_width
        )
        self.uncountable[0][5:].set_color(RED)
        VGroup(
            self.finite, self.countable, self.uncountable
        ).arrange(DOWN, buff=1, aligned_edge=LEFT).shift(RIGHT)
        self.countable_examples = MathTex(
            r"\mathbb{N},~\mathbb{Z},~\mathbb{Q},\cdots", color=BLUE,
            stroke_width=self.stroke_width, font_size=self.font_size
        ).move_to(self.countable).align_to(self.countable[0][4], LEFT)
        self.empty = Tex(
            r"（包括$\varnothing$）", tex_template=TexTemplateLibrary.ctex,
            stroke_width=2, font_size=40
        ).next_to(self.finite, buff=.5)
        self.at_most = Brace(VGroup(self.finite, self.countable), LEFT)
        self.at_most.label = Tex(
            "至多可数集", tex_template=TexTemplateLibrary.ctex,
            font_size=48, stroke_width=self.stroke_width
        ).next_to(self.at_most, LEFT)

    def construct(self):
        self.add(self.finite, self.empty)
        self.wait()
        self.play(Write(self.countable))
        self.wait()
        self.play(ReplacementTransform(self.countable[0][4:], self.countable_examples))
        self.wait()
        self.play(GrowFromCenter(self.at_most), GrowFromCenter(self.at_most.label))
        self.wait()
        self.play(Write(self.uncountable))
        self.wait()


class Countable(Scene):
    def __init__(self):
        super().__init__()
        self.question = MathTex(
            r"{{\varepsilon_1}} + {{\varepsilon_2}} + {{\varepsilon_3}} + \cdots = {{\varepsilon }}?",
            font_size=60, stroke_width=2
        ).to_edge(UP, buff=1).set_color_by_tex_to_color_map({'varepsilon_': BLUE, 'varepsilon ': YELLOW})
        self.geometric = MathTex(
            r"\tfrac12 + \tfrac14 + \tfrac18 + \cdots = 1",
            font_size=60, stroke_width=2
        ).next_to(self.question, DOWN, buff=.5)

        self.epsilon_n = MathTex(
            r"{{\varepsilon_n}} = {{{\varepsilon }}\over 2^n}",
            font_size=60, stroke_width=2
        ).next_to(self.geometric, DOWN, buff=.5).set_color_by_tex_to_color_map(
            {'varepsilon_': BLUE, 'varepsilon ': YELLOW}
        )
        # self.add(index_labels(self.epsilon_n[-1]))
        self.epsilon_n[-1][-1].set_color(BLUE)

        self.zero_examples = MathTex(
            r"m^*({{\mathbb{Z}}})=m^*({{\mathbb{N}}})=m^*({{\mathbb{Q}}})=0",
            font_size=60, stroke_width=2
        ).next_to(self.epsilon_n, DOWN, buff=.5).set_color_by_tex_to_color_map({'mathbb': YELLOW})
        self.conclusion = Text("至多可数集是零测集.").next_to(self.zero_examples, DOWN, buff=.5)
        self.conclusion[:5].set_color(YELLOW)

    def construct(self):
        self.add(self.question)
        self.wait()
        self.play(Write(self.geometric))
        self.wait()
        self.play(Write(self.epsilon_n))
        self.wait()
        self.play(Write(self.zero_examples))
        self.wait()
        self.play(Write(self.conclusion))

@cache
def cantor_set_recursive(n, a=0, b=1):
    """
    Compute the n-th Cantor set using recursion.

    Parameters:
        n (int): The step of the Cantor set to compute.
        a (float): The starting point of the current interval (default is 0).
        b (float): The ending point of the current interval (default is 1).

    Returns:
        List[Tuple[float, float]]: A list of tuples representing the intervals in the n-th Cantor set.
    """
    # Base case: at the 1st level, return the initial interval
    if n == 1:
        return [(a, b)]

    # Divide the interval into three parts
    length = (b - a) / 3
    left_third = cantor_set_recursive(n - 1, a, a + length)  # Left third
    right_third = cantor_set_recursive(n - 1, a + 2 * length, b)  # Right third

    # Combine the intervals from both sides
    return left_third + right_third


class CantorSet(Scene):
    @staticmethod
    def get_lines_from_intervals(intervals):
        factor = 10
        return VGroup(
            *[Line(
                (a * factor, 0, 0), (b * factor, 0, 0),
                stroke_width=12
            ) for (a, b) in intervals]
        )

    def construct(self):
        title = Title("Cantor Set")
        self.add(title)
        list_of_intervals = [cantor_set_recursive(n) for n in range(1, 11)]
        lines = VGroup(
            *[self.get_lines_from_intervals(intervals) for intervals in list_of_intervals]
        ).arrange(DOWN, buff=0.5).set_color(YELLOW).next_to(title, DOWN, buff=1)
        self.play(Create(lines[0]))
        for i in range(len(lines) - 1):
            self.play(FadeTransform(lines[i].copy(), lines[i+1]))
        self.wait()


class AlmostEverywhere(Scene):
    def __init__(self):
        super().__init__()
        self.title = Title("\\heiti 几乎处处（almost everywhere, a.e.）", tex_template=TexTemplateLibrary.ctex)
        self.meaning = Text(
            "一个命题在除去一个零测集后成立，称它几乎处处成立.",
            font_size=36
        ).next_to(self.title, DOWN, buff=.5)
        self.meaning[5:12].set_color(YELLOW)
        self.meaning[18:22].set_color(YELLOW)

        self.dirichlet = Tex(
            r"$D(x)="
            r"\begin{cases}"
            r"1,\quad x\in\mathbb{Q}\\"
            r"0,\quad x\notin\mathbb{Q}\\"
            r"\end{cases}$几乎处处为0.",
            tex_template=TexTemplateLibrary.ctex,
            font_size=60, stroke_width=2
        ).next_to(self.meaning, DOWN, 1)
        self.dirichlet[0][17:21].set_color(YELLOW)
        self.riemann = Tex(
            r"$[a,b]$上的有界$f$ Riemann可积$\iff ~f$几乎处处连续.",
            tex_template = TexTemplateLibrary.ctex,
            font_size = 50, stroke_width = 2
        ).next_to(self.dirichlet, DOWN, 1)
        self.riemann[0][22:26].set_color(YELLOW)

    def construct(self):
        self.add(self.title)
        self.wait()
        for text in (self.meaning, self.dirichlet, self.riemann):
            self.play(Write(text))
            self.wait()


class SubAdditive(Scene):
    def __init__(self):
        super().__init__()
        self.additive = MathTex(
            r"m({{E_1}} \cup {{E_2}} \cup \cdots) = m({{E_1}})+m({{E_2}})+\cdots",
            font_size=60, stroke_width=2
        ).set_color_by_tex_to_color_map({'E_': YELLOW}).to_edge(UP, buff=1)
        self.disjoint = Brace(
            VGroup(self.additive[1:4], self.additive[4][:4]), DOWN
        ).set_color(BLUE)
        self.disjoint.label = MathTex(r"E_i\cap E_j=\varnothing", color=BLUE).next_to(self.disjoint, DOWN)
        self.sub_additive = MathTex(
            r"m^*({{E_1}} \cup {{E_2}} \cup \cdots) \leq m^*({{E_1}})+m^*({{E_2}})+\cdots",
            font_size=60, stroke_width=2
        ).set_color_by_tex_to_color_map({'E_': YELLOW}).next_to(self.additive, DOWN, buff=2)
        self.arrow = Arrow(ORIGIN, UP, color=RED, stroke_width=12, buff=0).next_to(self.sub_additive[4][5], DOWN)
        self.arrow.label = Tex(
            "存在糟糕的$E_i$使得不等号严格成立",
            tex_template=TexTemplateLibrary.ctex,
            stroke_width=2,
            color=self.arrow.get_color()
        ).next_to(self.arrow, DOWN)

    def construct(self):
        self.play(Write(self.additive))
        self.play(GrowFromCenter(self.disjoint), GrowFromCenter(self.disjoint.label))
        self.wait()
        self.play(Write(self.sub_additive))
        self.wait()
        self.play(GrowArrow(self.arrow))
        self.wait()
        self.play(Write(self.arrow.label))
        self.wait()


class Solutions(Scene):
    def __init__(self):
        super().__init__()
        self.font_size = 60
        self.first = Text("1. 舍弃测度的某些良好性质.", font_size=self.font_size)
        self.second = Text("2. 舍弃那些糟糕的集合.", font_size=self.font_size)
        VGroup(self.first, self.second).arrange(DOWN, buff=1, aligned_edge=LEFT)

        self.line = Line(self.first.get_left(), self.first.get_right()).set_opacity(.2)

    def construct(self):
        self.play(Write(self.first))
        self.wait()
        self.play(Write(self.second))
        self.wait()
        self.play(
            Create(self.line), self.first.animate.set_opacity(.2)
        )
        self.wait()


class Caratheodory(Scene):
    def __init__(self):
        super().__init__()
        self.E = Ellipse(
            width=5, height=3, fill_color=YELLOW, fill_opacity=.2, stroke_color=YELLOW
        )#.shift(UP)
        self.E.label = MathTex("E", color=YELLOW, font_size=50).next_to(self.E, UP)

        self.tests = VGroup(
            Ellipse(
                fill_color=BLUE, fill_opacity=.2, stroke_color=BLUE, width=4, height=2
            ).move_to(self.E.get_left()),
            Rectangle(
                fill_color=BLUE, fill_opacity=.2, stroke_color=BLUE, width=4, height=2
            ).move_to(self.E.get_bottom()),
            Star(
                fill_color=BLUE, fill_opacity=.2, stroke_color=BLUE, outer_radius=2
            ).move_to(self.E.get_right())
        )
        self.inners = VGroup(*[
            Intersection(test, self.E, stroke_color=GREEN, fill_color=GREEN, fill_opacity=.6)
            for test in self.tests
        ])
        self.outers = VGroup(*[
            Difference(test, self.E, stroke_color=BLUE, fill_color=BLUE, fill_opacity=.6)
            for test in self.tests
        ])

        self.criterion_def = MathTex(
            r"m^*({{T}})=m^*({{T}}\cap {{E}})+m^*({{T}}\cap {{E^c}})",
            font_size=60, stroke_width=2
        ).next_to(self.E, DOWN, buff=1).set_color_by_tex_to_color_map({'T': BLUE, 'E': YELLOW})
        self.title = Title("\\heiti Carathéodory条件", font_size=50, tex_template=TexTemplateLibrary.ctex)

    def construct(self):
        self.add(self.E, self.E.label)
        self.wait()
        i = 0
        for test, inner, outer in zip(self.tests, self.inners, self.outers):
            self.add(test)
            self.wait()
            self.play(FadeIn(inner), FadeIn(outer))
            self.wait()
            if i < len(self.tests) - 1:
                self.remove(test, inner, outer)
            i += 1
        self.wait()
        self.play(Write(self.criterion_def))
        self.wait()
        self.play(Write(self.title))
        self.wait()


class Comparison(Scene):
    def __init__(self):
        super().__init__()
        self.outer = Tex(
            r"$m^*$定义在$\mathbb{R}$的任何子集上（$E\subseteq\mathbb{R}$）", tex_template=TexTemplateLibrary.ctex,
            font_size=60, stroke_width=2
        )
        self.measure = Tex(
            "$m$定义在$\mathbb{R}$的某些子集上（$E\subseteq\mathbb{R}$且可测）", tex_template=TexTemplateLibrary.ctex,
            font_size=60, stroke_width=2
        )
        VGroup(self.outer, self.measure).arrange(DOWN, buff=2.5).set_y(0.8)
        self.outer.align_to(self.measure, LEFT)
        self.outer_def = MathTex(
            r"m^*(E)=\inf\left\{\sum_{i=1}^\infty |I_i|\colon E\subseteq \bigcup_{i=1}^{\infty} I_i\right\}",
            font_size=40, stroke_width=1
        ).next_to(self.outer, DOWN, buff=.3).set_x(0)
        for e_idx in (3, 20):
            self.outer_def[0][e_idx].set_color(YELLOW)
        for i_idx in (slice(10, 19), slice(22, -1)):
            self.outer_def[0][i_idx].set_color(BLUE)
        self.measure_def = MathTex(
            r"m(E)=m^*(E)", substrings_to_isolate='E', font_size=40, stroke_width=1
        ).set_color_by_tex_to_color_map({'E': YELLOW}).next_to(self.measure, DOWN, buff=.5).set_x(0)

        self.outer[0][7:9].set_color(RED)
        VGroup(self.measure[0][6:8], self.measure[0][16:18]).set_color(RED)

        self.constant = MathTex("f^*(x)=1", color=RED, stroke_width=1).next_to(self.outer, DOWN, buff=.5, aligned_edge=LEFT)
        self.ln = MathTex(r"f(x)=\tfrac{\ln x}{\ln x}", color=RED, stroke_width=1).next_to(self.measure, DOWN, buff=.5, aligned_edge=LEFT)

    def construct(self):
        for t in (self.outer, self.outer_def, self.measure, self.measure_def, self.constant, self.ln):
            self.play(Write(t))
            self.wait()


class MeasureSpace(Scene):
    def __init__(self):
        super().__init__()
        self.title = Title(
            "\\heiti 测度空间",
            font_size=50, tex_template=TexTemplateLibrary.ctex,
        ).set_color(YELLOW)
        self.triplet = MathTex(
            r"({{X}}, {{\Sigma}}, {{\mu}})",
            font_size=120
        ).set_color_by_tex_to_color_map({'X': BLUE, 'Sigma': RED, 'mu': YELLOW})
        self.bottom_arrow = Arrow(ORIGIN, UP * .5, buff=0).next_to(self.triplet[3], DOWN)
        self.upper_arrow = Arrow(ORIGIN, DOWN * .5, buff=0).next_to(self.triplet[3], UP)
        self.some_subsets = Tex(
            "{{$X$}}的某些子集构成的集合",
            tex_template=TexTemplateLibrary.ctex,
            stroke_width=2
        ).set_color_by_tex('X', BLUE)
        self.sigma_algebra = Tex(
            r"满足某些性质，被称为{{$X$}}上的{{$\sigma$-代数}}",
            tex_template=TexTemplateLibrary.ctex,
            stroke_width=2
        ).set_color_by_tex_to_color_map({'X': BLUE, 'sigma': RED})
        self.scope = Tex(
            r"最小取{{$\{\varnothing, X\}$}}，最大取{{$2^X$}}",
            tex_template=TexTemplateLibrary.ctex,
            stroke_width=2
        ).next_to(self.upper_arrow, UP).set_color_by_tex("X", BLUE)
        VGroup(self.some_subsets, self.sigma_algebra).arrange(DOWN).next_to(self.bottom_arrow, DOWN)

    def construct(self):
        self.add(self.title)
        self.wait()
        self.play(Write(self.triplet))
        self.wait()
        self.play(GrowArrow(self.bottom_arrow))
        self.play(Write(self.some_subsets))
        self.wait()
        self.play(Write(self.sigma_algebra))
        self.wait()
        self.play(GrowArrow(self.upper_arrow))
        self.play(Write(self.scope))
        self.wait()


class ThumbNail(Scene):
    def __init__(self):
        super().__init__()
        self.curve = VGroup().set_points_smoothly(
            [UP, LEFT*3, LEFT*3+DOWN*2, DOWN*3.5, RIGHT*2+DOWN*2, RIGHT*3+DOWN, UP]
        ).shift(UP).set_style(
            fill_color=YELLOW, stroke_color=YELLOW, fill_opacity=.2, stroke_width=8
        )
        self.cover = VGroup(
            self.get_rectangle(4, 3, (-2.4, 0, 0)),
            self.get_rectangle(4, 2, (-0.5, 0.5, 0)),
            self.get_rectangle(3, 3, (1.7, 0.6, 0)),
            self.get_rectangle(2, 3, (1, -0.9, 0)),
            self.get_rectangle(1, 4, (0, -2.3, 0)),
        )
        self.title = MathTex(
            r"m(E)=m^*(E)", substrings_to_isolate='E', font_size=120, stroke_width=3
        ).set_color_by_tex_to_color_map({'E': YELLOW})
        VGroup(
            self.title,
            VGroup(self.curve, self.cover)
        ).arrange(DOWN).move_to(ORIGIN)

    def get_rectangle(self, height, width, center):
        return DashedVMobject(
            Rectangle(height=height, width=width, color=BLUE, stroke_width=8),
            dashed_ratio=0.5, num_dashes=70
        ).move_to(center)

    def construct(self):
        self.add(self.curve, self.cover, self.title)