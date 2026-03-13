from manim import *


Tex.set_default(stroke_width=5, tex_template=TexTemplateLibrary.ctex)
MathTex.set_default(stroke_width=5, tex_template=TexTemplateLibrary.ctex)
Title.set_default(tex_template=TexTemplateLibrary.ctex)



class CayleyHamilton(Scene):
    def construct(self):
        # 第一行：标题
        title = Tex(
            r"\textbf{哈密顿-凯莱定理}",
            font_size=120,
            color=BLUE
        )
        
        # 第二行：特征多项式
        char_poly = MathTex(
            r"p(\lambda) = \det(\lambda I - A)",
            color=WHITE
        )
        for i in (2, 9):
            char_poly[0][i].set_color(RED)
        char_poly = VGroup(Tex("其中"), char_poly).arrange().scale(1.3)
        # self.add(index_labels(char_poly[0]))
        
        # 第三行：核心公式
        formula = MathTex(
            r"p(A) = O",
            font_size=120,
            color=YELLOW
        )
        VGroup(title, formula, char_poly).arrange(DOWN, buff=1)
        
        self.add(title, char_poly, formula)

class MeanValueTheorem(Scene):
    def construct(self):
        # 坐标轴
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 6, 1],
            x_length=8,
            y_length=4,
            axis_config={"include_tip": False, "stroke_width": 2, "color": GREY},
            tips=False
        ).shift(DOWN * 1)
        
        # 曲线函数
        def f(x):
            return 0.5 * x**2 - 1.5 * x + 2.5
        
        curve = axes.plot(f, x_range=[0.1, 4.8], color=BLUE, stroke_width=5)
        
        # 端点
        x_a, x_b = 0.5, 4.5
        y_a, y_b = f(x_a), f(x_b)
        
        point_a = Dot(axes.c2p(x_a, y_a), color=RED, radius=0.12)
        point_b = Dot(axes.c2p(x_b, y_b), color=RED, radius=0.12)
        
        # 连接两端点的割线
        secant_line = Line(
            axes.c2p(x_a, y_a),
            axes.c2p(x_b, y_b),
            color=RED,
            stroke_width=8,
            # stroke_dasharray=[5, 5]
        )
        
        # 中值点c处的切线
        x_c = 2.5
        y_c = f(x_c)
        slope = 2 * 0.5 * x_c - 1.5  # f'(x) = x - 1.5
        
        point_c = Dot(axes.c2p(x_c, y_c), color=YELLOW, radius=0.12)
        
        # 在c处的切线
        tangent_x_range = [x_c - 1.5, x_c + 1.5]
        tangent_line = axes.plot(
            lambda x: y_c + slope * (x - x_c),
            x_range=tangent_x_range,
            color=YELLOW,
            stroke_width=10
        )
        tangent_line = DashedVMobject(tangent_line)
        
        # 标签
        label_a = MathTex("a", font_size=80, color=RED).next_to(point_a, DOWN, buff=0.3)
        label_b = MathTex("b", font_size=80, color=RED).next_to(point_b, DOWN, buff=0.3)
        label_c = MathTex("\\xi", font_size=80, color=YELLOW).next_to(point_c, DOWN, buff=0.3)
        title = Tex("\\textbf {微分中值定理}", font_size=120, color=WHITE).to_edge(UP, buff=1)
        
        self.add(
            axes,
            curve,
            secant_line,
            tangent_line,
            point_a,
            point_b,
            point_c,
            label_a,
            label_b,
            label_c,
            title
        )

class Heine(Scene):
    def construct(self):
        
        # 绘制连续函数曲线
        axes = Axes(
            x_range=[-1, 4, 1],
            y_range=[-0.5, 2, 1],
            x_length=10,
            y_length=8,
            axis_config={"include_tip": False, "stroke_width": 2, "stroke_opacity": 0.3},
        ).shift(DOWN * 0.5 + LEFT * .5)
        
        # 平滑的连续函数
        f = lambda x: 1.5 + 0.6 * np.sin(2 * x) + 0.3 * np.cos(3 * x) - 1
        func = axes.plot(
            f,
            x_range=[0, 3.5],
            color=BLUE,
            stroke_width=8
        )
        
        # 极限点
        limit_x = 2.5
        limit_y = f(limit_x)
        limit_point = Dot(axes.c2p(limit_x, limit_y), color=YELLOW, radius=0.15)
        # limit_glow = Circle(radius=0.25, color=YELLOW, stroke_width=6).move_to(limit_point)
        
        # 收敛的序列点
        sequence_points = VGroup()
        x_values = [1, 1.8, 2.1, 2.3, 2.42, 2.48]
        
        for i, x in enumerate(x_values):
            y = f(x)
            point = Dot(axes.c2p(x, y), color=RED, radius=0.12)
            sequence_points.add(point)

        arrow = CurvedArrow(sequence_points[0].get_center() + DOWN * .5, limit_point.get_center() + LEFT * .5)
            
        
        # 主标题 - 大号醒目
        title = Tex("\\heiti 海涅定理", font_size=150, color=WHITE)
        title.to_edge(UP, buff=1)
        
        # 副标题 - 核心概念
        subtitle = MathTex(
            r"\lim_{{{x}} \to {{a}}} ", "f", r"(x) = \lim_{n \to \infty}", "f", r"({{x_n}})",
            font_size=60,
            color=YELLOW
        ).next_to(title, DOWN)#.set_color_by_tex_to_color_map({'x_n': RED, 'f': BLUE, 'a': YELLOW, 'infty': WHITE}).to_edge(UP)
        
        self.add(axes, func)
        self.add(arrow)
        self.add(limit_point)
        # self.add(limit_glow, limit_point)
        self.add(sequence_points)
        self.add(title, subtitle)