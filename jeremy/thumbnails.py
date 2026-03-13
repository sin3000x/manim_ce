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


class Darboux(Scene):
    def construct(self):
        import matplotlib.pyplot as plt
        from PIL import Image
        import io
        
        # 创建 matplotlib 图形 - 透明背景
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='none')
        ax.set_facecolor('none')
        
        # 绘制导函数 f'(x) = 2x*sin(1/x) - cos(1/x)
        x_left = np.linspace(-3, -0.05, 1000)
        x_right = np.linspace(0.05, 3, 1000)
        
        def f_prime(x):
            return 2 * x * np.sin(1/x) - np.cos(1/x)
        
        y_left = f_prime(x_left)
        y_right = f_prime(x_right)
        
        ax.plot(x_left, y_left, color='#00BFFF', linewidth=3)
        ax.plot(x_right, y_right, color='#00BFFF', linewidth=3)
        
        # 在 x=0 处绘制高频震荡
        x_osc = np.linspace(-0.05, 0.05, 5000)
        y_osc = f_prime(x_osc)
        ax.plot(x_osc, y_osc, color='#00BFFF', linewidth=2)
        
        # 坐标轴设置 - 去掉网格、标签，保留坐标轴
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2, 2)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_position('zero')  # y轴在x=0处
        ax.spines['bottom'].set_position('zero')  # x轴在y=0处
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 保存为图像 - 透明背景
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='none', transparent=True)
        buf.seek(0)
        plt.close()
        
        # 转换为 manim 图像
        img = Image.open(buf)
        img_array = np.array(img)
        
        # 创建 manim 图像对象
        manim_img = ImageMobject(img_array).to_edge(DOWN, buff=-0.2)
        
        # 主标题
        title = Tex("\\textbf{达布定理}", font_size=180, color=WHITE).to_edge(UP, buff=0.6)
        
        # 核心内容
        content = VGroup(
            Tex("导函数具有介值性", font_size=50, color=YELLOW),
        ).arrange(DOWN, buff=0.2).next_to(title, DOWN, buff=0.5)
        
        self.add(title, content, manim_img.shift(DOWN * 0.8))


class Cantor(Scene):
    def construct(self):
        # 绘制闭区间上的连续函数
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[1, 4, 1],
            x_length=8,
            y_length=4,
            axis_config={"include_tip": False, "stroke_width": 3, "color": GREY},
        ).shift(DOWN * 0.5)
        
        # 平滑连续函数
        def f(x):
            return 2 + 0.8 * np.sin(2 * x) + 0.4 * np.cos(3 * x)
        
        curve = axes.plot(f, x_range=[0.5, 4.5], color=BLUE, stroke_width=6)
        
        # 端点标记
        a_point = Dot(axes.c2p(0.5, f(0.5)), color=RED, radius=0.15)
        b_point = Dot(axes.c2p(4.5, f(4.5)), color=RED, radius=0.15)
        
        a_label = MathTex("a", font_size=70, color=RED).next_to(a_point, DOWN, buff=0.3)
        b_label = MathTex("b", font_size=70, color=RED).next_to(b_point, DOWN, buff=0.3)
        
        # 展示一致连续性：任意两点距离小于δ时，函数值距离小于ε
        x1, x2 = 2.0, 2.6
        y1, y2 = f(x1), f(x2)
        
        point1 = Dot(axes.c2p(x1, y1), color=YELLOW, radius=0.12)
        point2 = Dot(axes.c2p(x2, y2), color=YELLOW, radius=0.12)
        
        # δ 和 ε 的虚线标记
        delta_line1 = DashedLine(axes.c2p(x1, 1), axes.c2p(x1, y1), color=YELLOW, stroke_width=4)
        delta_line2 = DashedLine(axes.c2p(x2, 1), axes.c2p(x2, y2), color=YELLOW, stroke_width=4)
        delta_label = MathTex(r"\delta", font_size=60, color=YELLOW).move_to(axes.c2p((x1+x2)/2, 0.5))
        
        epsilon_line1 = DashedLine(axes.c2p(0, y1), axes.c2p(x1, y1), color=GREEN, stroke_width=4)
        epsilon_line2 = DashedLine(axes.c2p(0, y2), axes.c2p(x2, y2), color=GREEN, stroke_width=4)
        epsilon_label = MathTex(r"\varepsilon", font_size=60, color=GREEN).move_to(axes.c2p(-0.3, (y1+y2)/2))
        
        # 主标题
        title = Tex("\\textbf{康托定理}", font_size=220, color=WHITE).to_edge(UP, buff=0.8)
        
        # 核心内容
        subtitle = Tex(
            "闭区间上连续函数必一致连续",
            font_size=55,
            color=YELLOW
        ).next_to(title, DOWN, buff=0.4)
        
        self.add(axes, curve)
        self.add(a_point, b_point, a_label, b_label)
        self.add(point1, point2)
        self.add(delta_line1, delta_line2, delta_label)
        self.add(epsilon_line1, epsilon_line2, epsilon_label)
        self.add(title)#, subtitle)