from manim import *
from pathlib import Path
from manim.scene.vector_space_scene import X_COLOR, Y_COLOR

# 让本文件内所有 Tex/MathTex 默认使用 ctex，无需逐个传 tex_template
config.tex_template = TexTemplateLibrary.ctex

# 设置全局默认 stroke_width 和 tex_template
Tex.set_default(stroke_width=2, tex_template=TexTemplateLibrary.ctex)
MathTex.set_default(stroke_width=2, tex_template=TexTemplateLibrary.ctex)
Title.set_default(tex_template=TexTemplateLibrary.ctex)

# 设置全局默认 Matrix h_buff
Matrix.set_default(h_buff=0.8)

class Opening(Scene):
    def construct(self):
        # 左侧：来自各个领域的方程
        equations = VGroup(
            # 数据科学：线性回归
            MathTex(r"\min_{\beta} ||X\beta - y||^2", font_size=36, color=BLUE),
            # 牛顿迭代
            MathTex(r"J(x_k)\Delta x = -F(x_k)", font_size=36, color=GREEN),
            # 微分方程：有限差分
            MathTex(r"\frac{u_{i+1} - 2u_i + u_{i-1}}{h^2} = f_i", font_size=36, color=RED),
            # 电路分析：基尔霍夫定律
            MathTex(r"R_1 i_1 + R_2 i_2 = V", font_size=36, color=PINK),
            # 网络分析：PageRank
            MathTex(r"(I - \alpha P^T)x = v", font_size=36, color=TEAL),
        ).arrange(DOWN, buff=0.8, aligned_edge=LEFT)
        equations.to_edge(LEFT, buff=2.5)
        
        # 领域标签
        labels = VGroup(
            Tex("数据科学", font_size=28),
            Tex("牛顿迭代", font_size=28),
            Tex("微分方程", font_size=28),
            Tex("电路分析", font_size=28),
            Tex("网络分析", font_size=28),
        )
        for i, label in enumerate(labels):
            label.next_to(equations[i], LEFT, buff=0.3).set_color(equations[i].get_color())
        
        # 中间：分析/转化的图案（齿轮）
        gear = VGroup()
        center_circle = Circle(radius=0.8, color=WHITE, fill_opacity=0.1)
        
        # 添加齿轮齿
        num_teeth = 16
        for i in range(num_teeth):
            angle = i * TAU / num_teeth
            tooth = Rectangle(width=0.3, height=0.15, color=WHITE, fill_opacity=0.3)
            tooth.move_to(center_circle.point_at_angle(angle) + 0.1 * np.array([np.cos(angle), np.sin(angle), 0]))
            tooth.rotate(angle)
            gear.add(tooth)
        
        gear.add(center_circle)
        inner_circle = Circle(radius=0.3, color=WHITE, fill_opacity=0.3)
        gear.add(inner_circle)
        
        transform_icon = gear
        transform_icon.move_to(ORIGIN).shift(RIGHT * 1.2)
        
        # 右侧：统一的 Ax = b 形式
        unified_form = MathTex(
            r"A", r"x", r"=", r"b",
            color=YELLOW,
            font_size=80
        )
        unified_form.to_edge(RIGHT, buff=1)
        # 强调统一形式
        highlight_box = SurroundingRectangle(
            unified_form,
            color=GOLD,
            buff=0.3,
            corner_radius=0.2
        )
        
        # 先显示齿轮
        self.play(
            FadeIn(transform_icon, scale=0.5),
            Write(unified_form),
            Create(highlight_box),
        )
        
        # 创建持续旋转的动画
        gear.add_updater(lambda m, dt: m.rotate(dt * 0.5))
        
        # 先显示 Ax=b
        self.wait(0.5)
        
        # 使用 LaggedStart 让所有标签和公式依次出现
        fade_in_anims = []
        for label, eq in zip(labels, equations):
            fade_in_anims.append(AnimationGroup(
                FadeIn(VGroup(label, eq), shift=RIGHT*0.3),
            ))
        
        self.play(
            LaggedStart(*fade_in_anims, lag_ratio=0.3),
            run_time=4
        )
        self.wait(0.5)
        
        # 创建所有方程的分身
        eq_copies = [eq.copy() for eq in equations]
        
        # 使用 LaggedStart 让所有分身依次缩入齿轮
        shrink_anims = []
        for eq_copy in eq_copies:
            shrink_anims.append(
                eq_copy.animate.scale(0.1).move_to(transform_icon.get_center())
            )
        
        self.play(
            LaggedStart(*shrink_anims, lag_ratio=0.2),
            run_time=3
        )
        
        # 移除所有分身
        for eq_copy in eq_copies:
            self.remove(eq_copy)
        
        # Ax=b 只闪烁一次
        self.play(
            Flash(unified_form, color=GOLD, flash_radius=1.5, line_length=0.4),
            run_time=0.6
        )
        
        self.wait(5)

ASSET_PATH = Path(__file__).parent / 'assets' / 'linear_system'
class Textbook(Scene):
    def construct(self):
        eola1 = ImageMobject(ASSET_PATH / '3b1b1.png').scale(.5)
        eola2 = ImageMobject(ASSET_PATH / '3b1b2.png').scale(.5)
        eola2.shift(DOWN * 1.5)
        eola = Group(eola2, eola1).to_edge(LEFT).to_edge(UP, buff=1)
        eola_label = Tex(r"《线性代数的本质 - 06》", color=YELLOW, font_size=40).next_to(eola, DOWN)

        ml = ImageMobject(ASSET_PATH / 'ml').scale(.6).to_edge(RIGHT, buff=1).to_edge(UP)
        ml_label = Tex(r"《机器学习》", color=YELLOW, font_size=40).next_to(ml, DOWN)

        rl = ImageMobject(ASSET_PATH / 'rl').scale(.6).next_to(ml_label, DOWN, buff=.5)
        rl_label = Tex("《动手学强化学习》", color=YELLOW, font_size=40).next_to(rl, DOWN)

        self.wait()

        self.add(eola, eola_label)
        self.wait(.5)

        self.add(ml, ml_label)
        self.wait(.5)

        self.add(rl, rl_label)
        self.wait()


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
        self.apply_matrix(self.A, run_time=2)
        self.play(Write(self.title))
        self.wait()

class ChangeBasis(LinearTransformationScene):
    def __init__(self, **kwargs):
        super().__init__(
            leave_ghost_vectors=False,
            show_basis_vectors=False,
            **kwargs
        )
        self.A = [[2, 2], [1, 3]]
        self.b = (-4, -1)
        self.x = np.linalg.solve(self.A, self.b)
        self.title = MathTex(
            r"A {{x}} = {{b}}",
            tex_to_color_map={'x': PINK, 'b': YELLOW},
            font_size=80
        ).to_edge(UP, buff=1).add_background_rectangle()
        self.title2 = MathTex(
            r"{{x}} = A^{-1}{{b}}",
            tex_to_color_map={'x': PINK, 'b': YELLOW},
            font_size=80
        ).to_edge(UP, buff=1).add_background_rectangle()

    def construct(self):
        self.add(self.title)
        
        # 创建 b_vec，但不使用 add_vector，这样它不会被变换
        b_vec = Vector(self.b, color=YELLOW)
        b_vec_label = (
            MathTex(r"b", color=YELLOW)
            .add_background_rectangle()
            .next_to(b_vec.get_end(), DL)
        )
        
        # 使用 add_vector 添加 x_vec，这样它会被变换
        x_vec_ghost = Vector(self.x, color=PINK).set_opacity(.3)
        self.add(x_vec_ghost)
        x_vec = self.add_vector(self.x, color=PINK)
        x_vec_label = (
            MathTex(r"x", color=PINK)
            .add_background_rectangle()
            .next_to(x_vec.get_end(), LEFT)
        )
        self.add(b_vec, b_vec_label, x_vec_label)

        self.apply_matrix(self.A, run_time=2)
        
        self.wait()
        self.apply_inverse(self.A, run_time=2)
        self.play(FadeTransform(self.title, self.title2))
        self.wait()

class ManualExample(Scene):
    def __init__(self):
        super().__init__()
        font_size = 40
        prob = VGroup(
            MathTex("2{{x_1}} + 1{{x_2}} + 1{{x_3}} + 0{{x_4}} = 1", font_size=font_size),
            MathTex("4{{x_1}} + 3{{x_2}} + 3{{x_3}} + 1{{x_4}} = 3", font_size=font_size),
            MathTex("8{{x_1}} + 7{{x_2}} + 9{{x_3}} + 5{{x_4}} = 11", font_size=font_size),
            MathTex("6{{x_1}} + 7{{x_2}} + 9{{x_3}} + 8{{x_4}} = 15", font_size=font_size),
        ).arrange(DOWN, aligned_edge=LEFT)
        for tex in prob:
            tex.set_color_by_tex_to_color_map({'x_1': BLUE, 'x_2': PINK, 'x_3': GREEN, 'x_4': RED})

        brace = Brace(prob, LEFT)
        self.prob = VGroup(prob, brace)

        self.matrix_form = Tex(r"，其中$A=\begin{bmatrix} 2 & 1 & 1 & 0 \\ 4 & 3 & 3 & 1 \\ 8 & 7 & 9 & 5 \\ 6 & 7 & 9 & 8 \end{bmatrix}$, $b=\begin{bmatrix} 1 \\ 3 \\ 11 \\ 15 \end{bmatrix}.$", font_size=font_size)

        self.mul = MathTex(
            r"A^{-1}b=\begin{bmatrix}"
            r"\tfrac{9}{4} & -\tfrac{3}{4} & -\tfrac{1}{4} & \tfrac{1}{4} \\[0.6em]"
            r"-3 & \tfrac{5}{2} & -\tfrac{1}{2} & 0 \\[0.6em]"
            r"-\tfrac{1}{2} & -1 & 1 & -\tfrac{1}{2} \\[0.6em]"
            r"\tfrac{3}{2} & -\tfrac{1}{2} & -\tfrac{1}{2} & \tfrac{1}{2}"
            r"\end{bmatrix} \begin{bmatrix} 1 \\[0.5em] 3 \\[0.5em] 11 \\[0.5em] 15 \end{bmatrix} = \begin{bmatrix} 1 \\[0.5em] -1 \\[0.5em] 0 \\[0.5em] 2 \end{bmatrix}",
        ).scale(.8).shift(1.2 * DOWN)
        self.mul[0][89:].set_color(YELLOW)
        
        VGroup(self.prob, self.matrix_form).arrange().to_edge(UP, buff=1)

        self.brace = Brace(self.mul[0][5:88], DOWN)
        self.n2 = Tex(f"matrix-vector multiplication: {{$O(n^2)$}} flops").set_color_by_tex_to_color_map({"n": YELLOW}).next_to(self.brace, DOWN, buff=.1)

    def construct(self):
        self.add(self.prob)
        self.play(Write(self.matrix_form))
        self.wait()
        self.play(Write(self.mul))
        self.wait()
        # self.add(index_labels(self.mul[0]))
        self.play(GrowFromCenter(self.brace))
        self.play(Write(self.n2))
        self.wait()


class Inverse(Scene):
    def __init__(self):
        super().__init__()        # [A | I] 增广矩阵
        font_size = 40
        cm = {'A': YELLOW, 'I': BLUE}
        self.aug_ai = MathTex(r"""
        \left[\begin{array}{@{}cccc|cccc@{}}
        2 & 1 & 1 & 0 & 1 & 0 & 0 & 0\\[0.6em]
        4 & 3 & 3 & 1 & 0 & 1 & 0 & 0\\[0.6em]
        8 & 7 & 9 & 5 & 0 & 0 & 1 & 0\\[0.6em]
        6 & 7 & 9 & 8 & 0 & 0 & 0 & 1
        \end{array}\right]
                             """, font_size=font_size)
        for tup in ((7, 11), (16, 20), (25, 29), (34, 38)):
            s = slice(*tup)
            self.aug_ai[0][s].set_color(YELLOW)
        for tup in ((12, 16), (21, 25), (30, 34), (39, 43)):
            s = slice(*tup)
            self.aug_ai[0][s].set_color(BLUE)

        
        # [I | A^-1] 增广矩阵
        self.aug_ia_inv = MathTex(r"""
        \left[\begin{array}{@{}cccc|cccc@{}}
        1 & 0 & 0 & 0 & \tfrac{9}{4} & -\tfrac{3}{4} & -\tfrac{1}{4} & \tfrac{1}{4}\\[0.6em]
        0 & 1 & 0 & 0 & -3 & \tfrac{5}{2} & -\tfrac{1}{2} & 0\\[0.6em]
        0 & 0 & 1 & 0 & -\tfrac{1}{2} & -1 & 1 & -\tfrac{1}{2}\\[0.6em]
        0 & 0 & 0 & 1 & \tfrac{3}{2} & -\tfrac{1}{2} & -\tfrac{1}{2} & \tfrac{1}{2}
        \end{array}\right]
                             """, font_size=font_size)

        for tup in ((7, 11), (26, 30), (41, 45), (57, 61)):
            s = slice(*tup)
            self.aug_ia_inv[0][s].set_color(BLUE)
        for tup in ((12, 26), (31, 41), (46, 57), (62, 76)):
            s = slice(*tup)
            self.aug_ia_inv[0][s].set_color(YELLOW)
        
        self.arrow1 = Arrow(LEFT, RIGHT)


        VGroup(self.aug_ai, self.arrow1, self.aug_ia_inv).arrange(buff=.25).shift(DOWN * .5)


        self.AI = MathTex(r"[{{A}} \mid {{I}}]", font_size=70, stroke_width=3).set_color_by_tex_to_color_map(cm).next_to(self.aug_ai, UP, buff=.75).shift(LEFT * .1)

        self.IA = MathTex(r"[{{I}} \mid {{A^{-1}}}]", font_size=70, stroke_width=3).set_color_by_tex_to_color_map(cm).next_to(self.aug_ia_inv, UP, buff=.75).shift(LEFT * .28)

        self.arrow2 = Arrow(self.AI.get_right(), self.IA.get_left(), buff=1) 

        self.row_reduction = Tex("行变换", font_size=40).next_to(self.arrow2, UP, buff=.1)

        self.flops = Tex("计算量：{{$2n^3$}} flops", stroke_width=3, font_size=50).set_color_by_tex_to_color_map({'n': YELLOW}).move_to(self.arrow1).shift(DOWN * 2.5)

    def construct(self) -> None:
        self.play(
            Write(self.AI), Write(self.aug_ai)
        )
        self.play(
            GrowArrow(self.arrow1), 
            GrowArrow(self.arrow2),
            Write(self.row_reduction)
        )
        self.play(
            Write(self.IA),
            Write(self.aug_ia_inv)
        )
        self.wait()
        self.play(
            Write(self.flops)
        )
        self.wait()

        # self.add(index_labels(self.aug_ai[0]))
        # self.add(index_labels(self.aug_ia_inv[0]))

class DirectMethod(Scene):
    def __init__(self):
        super().__init__()

    def construct(self):
        direct = VGroup(
            Tex("直接法", font_size=150, stroke_width=4),
            Tex("Direct Methods", font_size=60, stroke_width=4),
        ).arrange(DOWN).set_color(YELLOW)
        comment = Tex(r"（只涉及基础的线性代数，请放心食用\textasciitilde）", font_size=40)
        VGroup(direct, comment).arrange(DOWN, buff=1)

        self.add(comment)
        self.play(Write(direct))
        self.wait()

class GaussianElimination(Scene):
    def __init__(self):
        super().__init__()
        # [A | b] 增广矩阵
        font_size = 40
        cm = {'A': YELLOW, 'b': GREEN, 'I': BLUE}

        self.aug_ab = MathTex(r"""
        \left[\begin{array}{@{}cccc|c@{}}
        2 & 1 & 1 & 0 & 1\\[0.6em]
        4 & 3 & 3 & 1 & 3\\[0.6em]
        8 & 7 & 9 & 5 & 11\\[0.6em]
        6 & 7 & 9 & 8 & 15
        \end{array}\right]
                             """, font_size=font_size)
        self.aug_ab2 = MathTex(r"""
        \left[\begin{array}{@{}cccc|c@{}}
        2 & 1 & 1 & 0 & 0\\[0.6em]
        4 & 3 & 3 & 1 & 8\\[0.6em]
        8 & 7 & 9 & 5 & 12\\[0.6em]
        6 & 7 & 9 & 8 & 10
        \end{array}\right]
                             """, font_size=font_size)
        # 给 A 部分上色 (YELLOW)
        for tup in ((7, 11), (13, 17), (19, 23), (26, 30)):
            s = slice(*tup)
            self.aug_ab[0][s].set_color(YELLOW)
            self.aug_ab2[0][s].set_color(YELLOW)
        # 给 b 部分上色 (GREEN)
        for tup in ((12, 13), (18, 19), (24, 26), (31, 33)):
            s = slice(*tup)
            self.aug_ab[0][s].set_color(GREEN)
            self.aug_ab2[0][s].set_color(GREEN)

        
        # [I | A^-1b] 增广矩阵
        self.aug_ix = MathTex(r"""
        \left[\begin{array}{@{}cccc|c@{}}
        1 & 0 & 0 & 0 & 1\\[0.6em]
        0 & 1 & 0 & 0 & -1\\[0.6em]
        0 & 0 & 1 & 0 & 0\\[0.6em]
        0 & 0 & 0 & 1 & 2
        \end{array}\right]
                             """, font_size=font_size)
        self.aug_ix2 = MathTex(r"""
        \left[\begin{array}{@{}cccc|c@{}}
        1 & 0 & 0 & 0 & ?\\[0.6em]
        0 & 1 & 0 & 0 & ?\\[0.6em]
        0 & 0 & 1 & 0 & ?\\[0.6em]
        0 & 0 & 0 & 1 & ?
        \end{array}\right]
                             """, font_size=font_size)

        # 给 I 部分上色 (BLUE)
        for tup in ((7, 11), (13, 17), (20, 24), (26, 30)):
            s = slice(*tup)
            self.aug_ix[0][s].set_color(BLUE)
        # 给 x 部分上色 (GREEN)
        for tup in ((12, 13), (18, 20), (25, 26), (31, 32)):
            s = slice(*tup)
            self.aug_ix[0][s].set_color(GREEN)
        for tup in ((7, 11), (13, 17), (19, 23), (25, 29)):
            s = slice(*tup)
            self.aug_ix2[0][s].set_color(BLUE)
        for tup in ((12, 13), (18, 19), (24, 25), (30, 31)):
            s = slice(*tup)
            self.aug_ix2[0][s].set_color(GREEN)
        
        self.arrow1 = Arrow(LEFT, RIGHT)

        VGroup(self.aug_ab, self.arrow1, self.aug_ix).arrange(buff=.25).shift(DOWN * .5)

        self.aug_ab2.move_to(self.aug_ab)
        self.aug_ix2.move_to(self.aug_ix).align_to(self.aug_ix, LEFT)

        self.Ab = MathTex(r"[{{A}} \mid {{b}}]", font_size=70, stroke_width=3).set_color_by_tex_to_color_map(cm).next_to(self.aug_ab, UP, buff=.75)

        self.Ix = MathTex(r"[{{I}} \mid {{A^{-1}b}}]", font_size=70, stroke_width=3).set_color_by_tex_to_color_map(cm).next_to(self.aug_ix, UP, buff=.75)

        self.arrow2 = Arrow(self.Ab.get_right(), self.Ix.get_left(), buff=.5) 

        self.row_reduction = Tex("行变换", font_size=40).next_to(self.arrow2, UP, buff=.1)

        self.flops = Tex("计算量：{{$\\tfrac{2}{3}n^3$}} flops", stroke_width=3, font_size=50).set_color_by_tex_to_color_map({'n': YELLOW}).move_to(self.arrow1).shift(DOWN * 2.5)

    def construct(self) -> None:
        self.play(
            Write(self.Ab), Write(self.aug_ab)
        )
        self.play(
            GrowArrow(self.arrow1), 
            GrowArrow(self.arrow2),
            Write(self.row_reduction)
        )
        self.play(
            Write(self.Ix),
            Write(self.aug_ix)
        )
        self.wait()
        self.play(
            Write(self.flops)
        )
        self.wait()
        self.play(
            FadeTransform(self.aug_ab, self.aug_ab2),
            FadeTransform(self.aug_ix, self.aug_ix2),
        )

        # self.add(index_labels(self.aug_ab[0]))
        # self.add(index_labels(self.aug_ix2[0]))


class LU1(Scene):
    def __init__(self):
        super().__init__()
        self.a = np.array(
            [[2, 1, 1, 0],
            [4, 3, 3, 1],
            [8, 7, 9, 5],
            [6, 7, 9, 8]]
        )
        self.A = Matrix(self.a)
        self.l1A = Matrix([
            [2, 1, 1, 0],
            ['', 1, 1, 1],
            ['', 3, 5, 5],
            ['', 4, 6, 8]
        ])
        self.l1 = Matrix([
            [1, '', '', ''],
            [-2, 1, '', ''],
            [-4, '', 1, ''],
            [-3, '', '', 1]
        ]).set_color(BLUE)
        
        self.arrow = Arrow(LEFT, RIGHT)
        self.arrow_label = Tex("行倍加")
        VGroup(self.A, self.arrow, self.l1A).arrange(buff=0.5)       # 标签
        self.arrow_label = Tex("行倍加").next_to(self.arrow, UP)
        self.A_label = MathTex("A", font_size=50).next_to(self.A, UP, buff=0.3)

    def construct(self):
        # 显示原矩阵 A
        self.play(Write(self.A_label), Write(self.A))
        
        r1, r2, r3, r4 = self.A.get_rows()
        brackets = self.l1A.get_brackets()
        l1A_r1, l1A_r2, l1A_r3, l1A_r4 = self.l1A.get_rows()

        self.play(GrowArrow(self.arrow), Write(self.arrow_label))
        self.play(Write(brackets), TransformFromCopy(r1, l1A_r1), r1.animate.set_color(YELLOW), l1A_r1.animate.set_color(YELLOW))
        self.play(TransformFromCopy(VGroup(r1, r2), l1A_r2))
        self.play(TransformFromCopy(VGroup(r1, r3), l1A_r3))
        self.play(TransformFromCopy(VGroup(r1, r4), l1A_r4))
        self.wait()

        self.play(
            VGroup(self.A, self.A_label, self.arrow, self.arrow_label, self.l1A).animate.scale(.6).to_edge(UP, buff=.5)
        )
        v = VGroup(
            self.l1, MathTex("A="), Matrix([
            [2, 1, 1, 0],
            ['', 1, 1, 1],
            ['', 3, 5, 5],
            ['', 4, 6, 8]
        ])
        ).arrange().shift(DOWN)
        l1_label = MathTex("L_1").match_color(self.l1).next_to(self.l1, DOWN)
        self.play(Write(v))
        self.play(Write(l1_label))
        self.wait()

        v1 = VGroup(
            MathTex("L1").match_color(self.l1), MathTex("A="), Matrix([
            [2, 1, 1, 0],
            ['', 1, 1, 1],
            ['', 3, 5, 5],
            ['', 4, 6, 8]
        ])
        ).arrange().scale(.6).to_edge(UP, buff=.5)
        self.play(
            FadeOut(VGroup(self.A, self.A_label, self.arrow, self.arrow_label, self.l1A, l1_label)),
            ReplacementTransform(v, v1)
        )

        v2 = VGroup(
            Matrix([
                [1, '', '', ''],
                ['', 1, '', ''],
                ['', -3, 1, ''],
                ['', -4, '', 1]
            ]).set_color(BLUE),
            Matrix([
                [2, 1, 1, 0],
                ['', 1, 1, 1],
                ['', 3, 5, 5],
                ['', 4, 6, 8]
            ]),
            MathTex("="),
            Matrix([
                [2, 1, 1, 0],
                ['', 1, 1, 1],
                ['', '', 2, 2],
                ['', '', 2, 4]
            ])
        ).arrange().shift(DOWN)
        v2[1].get_rows()[1].set_color(YELLOW)
        v2[3].get_rows()[1].set_color(YELLOW)
        l2_label = MathTex("L_2", stroke_color=BLUE).next_to(v2[0], DOWN)
        self.play(Write(v2))
        self.play(Write(l2_label))


class LU2(Scene):
    def __init__(self):
        super().__init__()
        self.chain = VGroup(
            MathTex("{{L_3L_2L_1}}{{A}}=").set_color_by_tex_to_color_map({"L": BLUE, "A": YELLOW}),
            Matrix([
                [2, 1, 1, 0],
                ['', 1, 1, 1],
                ['', '', 2, 2],
                ['', '', '', 2]
            ]).set_color(GREEN).scale(.8),
            MathTex(r"~\Longrightarrow~{{A}}={{L_1^{-1}L_2^{-1}L_3^{-1}}}{{R}}").set_color_by_tex_to_color_map({"L": BLUE, "R": GREEN, "A": YELLOW, "arrow": WHITE}),
        ).arrange().to_edge(UP, buff=.5)

        # A 矩阵 (黄色)
        mat_A = Matrix([
            [2, 1, 1, 0],
            [4, 3, 3, 1],
            [8, 7, 9, 5],
            [6, 7, 9, 8]
        ])
        mat_A.set_color(YELLOW)
        
        # L 矩阵 (下三角，蓝色)
        mat_L = Matrix([
            [1, '', '', ''],
            [2, 1, '', ''],
            [4, 3, 1, ''],
            [3, 4, 1, 1]
        ])
        mat_L.set_color(BLUE)
        
        # U 矩阵 (上三角，绿色)
        mat_U = Matrix([
            [2, 1, 1, 0],
            ['', 1, 1, 1],
            ['', '', 2, 2],
            ['', '', '', 2]
        ])
        mat_U.set_color(GREEN)
        
        self.factor = VGroup(
            mat_A,
            MathTex("="),
            mat_L,
            mat_U
        ).arrange().scale(.8).next_to(self.chain, DOWN, buff=.5)
        
        # 添加标签
        self.labels = VGroup(
            MathTex("A").set_color(YELLOW).next_to(mat_A, DOWN, buff=0.3),
            MathTex("L").set_color(BLUE).next_to(mat_L, DOWN, buff=0.3),
            MathTex("U").set_color(GREEN).next_to(mat_U, DOWN, buff=0.3)
        )

    def construct(self):
        self.add(self.chain[:2])
        self.wait()
        self.play(Write(self.chain[2:]))

        self.wait()
        self.play(Write(self.factor))
        self.play(FadeIn(self.labels))
        self.wait()


class LU3(Scene):
    def __init__(self):
        super().__init__()
        # 第一行：L1^-1 的表示
        # 创建左边的矩阵和上标
        mat_l1 = Matrix([
            [1, '', '', ''],
            [-2, 1, '', ''],
            [-4, '', 1, ''],
            [-3, '', '', 1]
        ]).scale(.7)
        # 染色：-2黄色，-4蓝色，-3红色
        mat_l1.get_entries()[4].set_color(YELLOW)  # -2
        mat_l1.get_entries()[8].set_color(BLUE)    # -4
        mat_l1.get_entries()[12].set_color(RED)    # -3
        
        inv_sup1 = MathTex("^{-1}").scale(.7).next_to(mat_l1, UR, buff=0.05).shift(DOWN * 0.1)
        
        # 创建右边的矩阵
        mat_l1_inv = Matrix([
            [1, '', '', ''],
            [2, 1, '', ''],
            [4, '', 1, ''],
            [3, '', '', 1]
        ]).scale(.7)
        # 染色：2黄色，4蓝色，3红色
        mat_l1_inv.get_entries()[4].set_color(YELLOW)  # 2
        mat_l1_inv.get_entries()[8].set_color(BLUE)    # 4
        mat_l1_inv.get_entries()[12].set_color(RED)    # 3
        
        self.inv_line1 = VGroup(
            VGroup(mat_l1, inv_sup1),
            MathTex("="),
            mat_l1_inv
        ).arrange()
        
        # 添加L1标签
        # self.l1_label = MathTex("L_1").next_to(mat_l1, UP, buff=0.2)

        # 第二行：L1^-1 L2^-1 L3^-1 = L
        # 统一的缩放因子
        line2_scale = 0.7
        
        # 创建三个矩阵（不带上标）
        mat1 = Matrix([
            [1, '', '', ''],
            [2, 1, '', ''],
            [4, '', 1, ''],
            [3, '', '', 1]
        ]).scale(line2_scale)
        # mat1染色：2黄色，4蓝色，3红色
        mat1.get_entries()[4].set_color(YELLOW)   # 2
        mat1.get_entries()[8].set_color(BLUE)     # 4
        mat1.get_entries()[12].set_color(RED)     # 3
        
        mat2 = Matrix([
            [1, '', '', ''],
            ['', 1, '', ''],
            ['', 3, 1, ''],
            ['', 4, '', 1]
        ]).scale(line2_scale)
        # mat2染色：3绿色，4粉色
        mat2.get_entries()[9].set_color(GREEN)    # 3
        mat2.get_entries()[13].set_color(PINK)    # 4
        
        mat3 = Matrix([
            [1, '', '', ''],
            ['', 1, '', ''],
            ['', '', 1, ''],
            ['', '', 1, 1]
        ]).scale(line2_scale)
        # mat3染色：1金色
        mat3.get_entries()[14].set_color(GOLD)    # 1
        
        mat_L = Matrix([
            [1, '', '', ''],
            [2, 1, '', ''],
            [4, 3, 1, ''],
            [3, 4, 1, 1]
        ]).scale(line2_scale)
        # mat_L染色：对应元素相同颜色
        mat_L.get_entries()[4].set_color(YELLOW)   # 2
        mat_L.get_entries()[8].set_color(BLUE)     # 4
        mat_L.get_entries()[9].set_color(GREEN)    # 3
        mat_L.get_entries()[12].set_color(RED)     # 3
        mat_L.get_entries()[13].set_color(PINK)    # 4
        mat_L.get_entries()[14].set_color(GOLD)    # 1
        
        self.inv_line2 = VGroup(
            mat1,
            mat2,
            mat3,
            MathTex("="),
            mat_L
        ).arrange(buff=0.3)

        # 布局
        self.inv_line1.to_edge(UP, buff=.5)
        # self.l1_label.next_to(mat_l1, UP, buff=0.2)
        self.inv_line2.next_to(self.inv_line1, DOWN, buff=1.2)
        
        # 添加逆矩阵标签（在矩阵下方）
        self.inv_labels = VGroup(
            MathTex("L_1^{-1}").scale(.8).next_to(mat1, DOWN, buff=0.3),
            MathTex("L_2^{-1}").scale(.8).next_to(mat2, DOWN, buff=0.3),
            MathTex("L_3^{-1}").scale(.8).next_to(mat3, DOWN, buff=0.3),
            MathTex("L").scale(.8).next_to(mat_L, DOWN, buff=0.3)
        )


    def construct(self):
        # 第一行动画
        # 获取左右矩阵的组件
        mat_l1 = self.inv_line1[0][0]  # 左边的矩阵
        inv_sup1 = self.inv_line1[0][1]  # 上标
        equals1 = self.inv_line1[1]  # 等号
        mat_l1_inv = self.inv_line1[2]  # 右边的矩阵
        
        # 先显示左边矩阵的括号和对角线1
        left_brackets = mat_l1.get_brackets()
        left_diag = VGroup(mat_l1.get_entries()[0], mat_l1.get_entries()[5], 
                          mat_l1.get_entries()[10], mat_l1.get_entries()[15])
        left_colored = VGroup(mat_l1.get_entries()[4], mat_l1.get_entries()[8], 
                             mat_l1.get_entries()[12])  # -2, -4, -3
        
        # 显示右边矩阵的括号和对角线1
        right_brackets = mat_l1_inv.get_brackets()
        right_diag = VGroup(mat_l1_inv.get_entries()[0], mat_l1_inv.get_entries()[5],
                           mat_l1_inv.get_entries()[10], mat_l1_inv.get_entries()[15])
        
        self.play(
            Write(left_brackets),
            Write(left_diag),
            Write(left_colored),  # 显示左边的彩色元素
            Write(inv_sup1),
            Write(equals1),
            Write(right_brackets),
            Write(right_diag)
        )
        
        # 把左边的-2、-4、-3 TransformFromCopy到右边对应元素（用lagged start）
        transforms = [
            TransformFromCopy(mat_l1.get_entries()[4], mat_l1_inv.get_entries()[4]),   # -2 -> 2
            TransformFromCopy(mat_l1.get_entries()[8], mat_l1_inv.get_entries()[8]),   # -4 -> 4
            TransformFromCopy(mat_l1.get_entries()[12], mat_l1_inv.get_entries()[12])  # -3 -> 3
        ]
        self.play(LaggedStart(*transforms, lag_ratio=0.3))
        self.wait()
        
        # 第二行动画
        # 获取矩阵组件
        mat1 = self.inv_line2[0]
        mat2 = self.inv_line2[1]
        mat3 = self.inv_line2[2]
        equals2 = self.inv_line2[3]
        mat_L = self.inv_line2[4]
        
        # 显示所有矩阵的括号和对角线1
        mat1_brackets = mat1.get_brackets()
        mat1_diag = VGroup(mat1.get_entries()[0], mat1.get_entries()[5],
                          mat1.get_entries()[10], mat1.get_entries()[15])
        mat1_colored = VGroup(mat1.get_entries()[4], mat1.get_entries()[8], 
                             mat1.get_entries()[12])  # 2, 4, 3
        
        mat2_brackets = mat2.get_brackets()
        mat2_diag = VGroup(mat2.get_entries()[0], mat2.get_entries()[5],
                          mat2.get_entries()[10], mat2.get_entries()[15])
        mat2_colored = VGroup(mat2.get_entries()[9], mat2.get_entries()[13])  # 3, 4
        
        mat3_brackets = mat3.get_brackets()
        mat3_diag = VGroup(mat3.get_entries()[0], mat3.get_entries()[5],
                          mat3.get_entries()[10], mat3.get_entries()[15])
        mat3_colored = VGroup(mat3.get_entries()[14])  # 1
        
        matL_brackets = mat_L.get_brackets()
        matL_diag = VGroup(mat_L.get_entries()[0], mat_L.get_entries()[5],
                          mat_L.get_entries()[10], mat_L.get_entries()[15])
        
        self.play(
            Write(mat1_brackets), Write(mat1_diag), Write(mat1_colored),
            Write(mat2_brackets), Write(mat2_diag), Write(mat2_colored),
            Write(mat3_brackets), Write(mat3_diag), Write(mat3_colored),
            Write(equals2),
            Write(matL_brackets), Write(matL_diag),
            Write(self.inv_labels)
        )
        
        # 把左边矩阵中的元素TransformFromCopy到右边结果中（相同颜色对应）
        # mat1: 2(黄), 4(蓝), 3(红) -> mat_L: 2(黄), 4(蓝), 3(红)
        # mat2: 3(绿), 4(粉) -> mat_L: 3(绿), 4(粉)
        # mat3: 1(金) -> mat_L: 1(金)
        color_transforms = [
            TransformFromCopy(mat1.get_entries()[4], mat_L.get_entries()[4]),    # 2黄
            TransformFromCopy(mat1.get_entries()[8], mat_L.get_entries()[8]),    # 4蓝
            TransformFromCopy(mat1.get_entries()[12], mat_L.get_entries()[12]),  # 3红
            TransformFromCopy(mat2.get_entries()[9], mat_L.get_entries()[9]),    # 3绿
            TransformFromCopy(mat2.get_entries()[13], mat_L.get_entries()[13]),  # 4粉
            TransformFromCopy(mat3.get_entries()[14], mat_L.get_entries()[14])   # 1金
        ]
        self.play(LaggedStart(*color_transforms, lag_ratio=0.2))
        self.wait()


class TriangularSystem(Scene):
    def __init__(self):
        super().__init__()
        
        # L 矩阵 (蓝色)
        mat_L = Matrix([
            [1, '', '', ''],
            [2, 1, '', ''],
            [4, 3, 1, ''],
            [3, 4, 1, 1]
        ]).set_color(BLUE)
        
        # U 矩阵 (绿色)
        mat_U = Matrix([
            [2, 1, 1, 0],
            ['', 1, 1, 1],
            ['', '', 2, 2],
            ['', '', '', 2]
        ]).set_color(GREEN)
        
        # 向量 x (黄色)
        vec_x = Matrix([[r'x_1'], [r'x_2'], [r'x_3'], [r'x_4']]).set_color(YELLOW)
        
        # 向量 y (红色)
        vec_y = Matrix([[r'y_1'], [r'y_2'], [r'y_3'], [r'y_4']]).set_color(RED)
        
        # 向量 b (居中对齐)
        vec_b = Matrix([[1], [3], [11], [15]], element_alignment_corner=ORIGIN)
        
        # 第一行：LUx = b
        self.line1 = VGroup(
            mat_L.copy(),
            mat_U.copy(),
            vec_x.copy(),
            MathTex("="),
            vec_b.copy()
        ).arrange(buff=0.2).scale(0.7)
        
        # 第二行：Ly = b
        self.line2 = VGroup(
            mat_L.copy(),
            vec_y.copy(),
            MathTex("="),
            vec_b.copy()
        ).arrange(buff=0.2).scale(0.5)
        
        # 第三行：Ux = y
        self.line3 = VGroup(
            mat_U.copy(),
            vec_x.copy(),
            MathTex("="),
            vec_y.copy()
        ).arrange(buff=0.2).scale(0.5)
        
        # 布局
        self.line1.to_edge(UP, buff=0.75)
        
        # 第二行和第三行组合
        self.lines_23 = VGroup(self.line2, self.line3).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        self.lines_23.next_to(self.line1, DOWN, buff=0.7).shift(LEFT)
        
        # 添加左侧的brace
        self.brace = Brace(self.lines_23, LEFT, buff=.5, sharpness=1)

        self.flops = Tex("$O(n^2)$ flops").next_to(self.brace, LEFT, buff=.5)
        self.lu_flops = Tex(r"LU factorization: \\[1em]$\dfrac{2}{3}n^3$ flops", font_size=55).next_to(self.lines_23, RIGHT, buff=1)

    def construct(self):
        # 创建独立的标签，每个字母对齐到对应的矩阵/向量
        offset = 1.5
        label_L = MathTex("L").set_color(BLUE).move_to(self.line1[0].get_center() + UP * offset)
        label_U = MathTex("U").set_color(GREEN).move_to(self.line1[1].get_center() + UP * offset)
        label_x = MathTex("x").set_color(YELLOW).move_to(self.line1[2].get_center() + UP * offset)
        label_eq = MathTex("=").move_to(self.line1[3].get_center() + UP * offset)
        label_b = MathTex("b").move_to(self.line1[4].get_center() + UP * offset)
        
        labels = VGroup(label_L, label_U, label_x, label_eq, label_b)
        self.play(Write(self.line1), Write(labels))
        
        self.wait()
        
        # 显示第二行和第三行以及brace
        self.play(
            GrowFromCenter(self.brace),
            Write(self.line2),
            Write(self.line3)
        )
        self.wait()

        self.play(Write(self.flops))
        self.wait()

        self.play(Write(self.lu_flops))
        self.wait()


class PartialPivoting1(Scene):
    def __init__(self):
        super().__init__()
        
        # A 矩阵 (黄色)
        mat_A = Matrix([
            [2, 1, 1, 0],
            [4, 3, 3, 1],
            [8, 7, 9, 5],
            [6, 7, 9, 8]
        ])
        
        # L 矩阵 (下三角，蓝色) - 用 * 表示非零元素
        mat_L = Matrix([
            ['*', '', '', ''],
            ['*', '*', '', ''],
            ['*', '*', '*', ''],
            ['*', '*', '*', '*']
        ]).set_color(BLUE)
        
        # U 矩阵 (上三角，绿色) - 用 * 表示非零元素
        mat_U = Matrix([
            ['*', '*', '*', '*'],
            ['', '*', '*', '*'],
            ['', '', '*', '*'],
            ['', '', '', '*']
        ]).set_color(GREEN)
        
        # 组合：A = L × U
        self.decomposition = VGroup(
            mat_A,
            MathTex("="),
            mat_L,
            mat_U
        ).arrange(buff=0.3)
        
        # 创建四个方框，框住 A 矩阵的顺序主子式
        # 获取 A 矩阵的元素
        entries = mat_A.get_entries()
        
        # 1x1 主子式：第一个元素
        self.box1 = SurroundingRectangle(entries[0], buff=0.15, stroke_width=3)
        
        # 2x2 主子式：前两行两列
        group_2x2 = VGroup(entries[0], entries[1], entries[4], entries[5])
        self.box2 = SurroundingRectangle(group_2x2, buff=0.15, stroke_width=3)
        
        # 3x3 主子式：前三行三列
        group_3x3 = VGroup(
            entries[0], entries[1], entries[2],
            entries[4], entries[5], entries[6],
            entries[8], entries[9], entries[10]
        )
        self.box3 = SurroundingRectangle(group_3x3, buff=0.15, stroke_width=3)
        
        # 4x4 主子式：整个矩阵
        self.box4 = SurroundingRectangle(mat_A.get_entries(), buff=0.15, stroke_width=3)
        
        # 将四个方框组合并设置从黄到红的渐变颜色
        self.boxes = VGroup(self.box1, self.box2, self.box3, self.box4)
        self.boxes.set_submobject_colors_by_gradient(YELLOW, RED)

    def construct(self):
        # 显示 A 矩阵
        self.add(self.decomposition)
        self.wait()
        
        # 依次显示四个顺序主子式的方框
        self.play(Create(self.box1))
        self.play(Create(self.box2))
        self.play(Create(self.box3))
        self.play(Create(self.box4))
        self.wait()
        
        # 淡出所有方框
        # self.play(FadeOut(VGroup(self.box1, self.box2, self.box3, self.box4)))
        # self.wait()


class PartialPivoting2(Scene):
    def __init__(self):
        super().__init__()
        
        # A 矩阵
        mat_A = Matrix([
            [0, 1],
            [1, 1]
        ])
        
        # L 矩阵 (下三角，蓝色) - 用 * 表示非零元素
        mat_L = Matrix([
            ['*', ''],
            ['*', '*']
        ]).set_color(BLUE)
        
        # U 矩阵 (上三角，绿色) - 用 * 表示非零元素
        mat_U = Matrix([
            ['*', '*'],
            ['', '*']
        ]).set_color(GREEN)
        
        # 第一行：A ≠ L × U
        self.line1 = VGroup(
            mat_A,
            MathTex(r"\neq"),
            mat_L,
            mat_U
        ).arrange(buff=0.3)
        
        # 交换后的矩阵
        mat_A_swapped = Matrix([
            [1, 1],
            [0, 1]
        ])
        
        # 置换矩阵 P
        mat_P = Matrix([
            [0, 1],
            [1, 0]
        ]).set_color(YELLOW)
        
        # PA = 交换后的结果
        self.line2 = VGroup(
            mat_P,
            mat_A.copy(),
            MathTex("="),
            mat_A_swapped,
            MathTex("={{L}}{{U}}").set_color_by_tex_to_color_map({'L': BLUE, 'U': GREEN})
        ).arrange(buff=0.3)
        
        # 布局
        self.line1.to_edge(UP, buff=1)
        self.line2.next_to(self.line1, DOWN, buff=1.2)
        
        # 添加标签
        self.label_P = MathTex("P").set_color(YELLOW).next_to(self.line2[0], UP, buff=0.3)

        self.partial_pivot = Tex("{{Partial}} Pivoting", font_size=60).set_color(YELLOW).to_edge(DOWN, buff=1)

    def construct(self):
        # 显示第一行：A ≠ L × U
        self.play(Write(self.line1))
        self.wait()
        
        # 获取 A 矩阵的两行
        mat_A = self.line1[0]

        row1, row2 = mat_A.get_rows()
        pos1, pos2 = row1.get_center(), row2.get_center()
        
        # 行交换动画：两行互换位置
        self.play(
            row1.animate.move_to(pos2),
            row2.animate.move_to(pos1),
        )
        
        # 换回来
        self.play(
            row1.animate.move_to(pos1),
            row2.animate.move_to(pos2),
        )
        self.wait()
        
        # 显示第二行：P 和 PA = 交换后的结果
        self.play(Write(self.line2), Write(self.label_P))
        self.wait()

        self.play(Write(self.partial_pivot))
        self.wait()

        self.play(Circumscribe(self.partial_pivot[0]))
        self.play(Circumscribe(self.partial_pivot[0]))
        self.wait()


class LUSequentialDependency(Scene):
    """展示 LU 分解的顺序依赖性，说明为什么不能很好地并行化"""
    
    def construct(self):
        # 创建一个 4x4 矩阵
        mat = Matrix([
            [2, 1, 1, 0],
            [4, 3, 3, 1],
            [8, 7, 9, 5],
            [6, 7, 9, 8]
        ])
        
        self.add(mat)
        
        # 获取矩阵元素
        entries = mat.get_entries()
        
        # 第一步：高亮主元和第一列
        pivot1 = SurroundingRectangle(entries[0], color=YELLOW, buff=0.1, stroke_width=4)
        col1_rects = VGroup(*[
            SurroundingRectangle(entries[i], color=YELLOW, buff=0.1, stroke_width=2)
            for i in [4, 8, 12]
        ])
        
        self.play(Create(pivot1), run_time=0.3)
        self.play(LaggedStart(*[Create(rect) for rect in col1_rects], lag_ratio=0.15), run_time=0.5)
        self.wait(0.3)
        
        # 第二步：高亮第二个主元和第二列
        pivot2 = SurroundingRectangle(entries[5], color=GREEN, buff=0.1, stroke_width=4)
        col2_rects = VGroup(*[
            SurroundingRectangle(entries[i], color=GREEN, buff=0.1, stroke_width=2)
            for i in [9, 13]
        ])
        
        self.play(
            pivot1.animate.set_stroke(opacity=0.3),
            col1_rects.animate.set_stroke(opacity=0.3),
            Create(pivot2),
            run_time=0.4
        )
        self.play(LaggedStart(*[Create(rect) for rect in col2_rects], lag_ratio=0.15), run_time=0.4)
        self.wait(0.3)
        
        # 第三步：高亮第三个主元和第三列
        pivot3 = SurroundingRectangle(entries[10], color=BLUE, buff=0.1, stroke_width=4)
        col3_rect = SurroundingRectangle(entries[14], color=BLUE, buff=0.1, stroke_width=2)
        
        self.play(
            pivot2.animate.set_stroke(opacity=0.3),
            col2_rects.animate.set_stroke(opacity=0.3),
            Create(pivot3),
            run_time=0.4
        )
        self.play(Create(col3_rect), run_time=0.3)
        self.wait(0.3)
        
        # 第四步：高亮最后一个主元
        pivot4 = SurroundingRectangle(entries[15], color=RED, buff=0.1, stroke_width=4)
        
        self.play(
            pivot3.animate.set_stroke(opacity=0.3),
            col3_rect.animate.set_stroke(opacity=0.3),
            Create(pivot4),
            run_time=0.4
        )
        self.wait()


class LUFillIn(Scene):
    """展示 LU 分解的 fill-in 现象：稀疏矩阵分解后变稠密"""
    
    def construct(self):
        # 创建一个箭头矩阵（arrow matrix）- 第一行和第一列有元素，其余是对角矩阵
        mat_A = Matrix([
            [1, 1, 1, 1, 1, 1],
            [1, 1, '', '', '', ''],
            [1, '', 1, '', '', ''],
            [1, '', '', 1, '', ''],
            [1, '', '', '', 1, ''],
            [1, '', '', '', '', 1]
        ], h_buff=0.9).scale(0.65)
        
        # L 矩阵（下三角）- 完全稠密！
        mat_L = Matrix([
            [1, '', '', '', '', ''],
            ['*', 1, '', '', '', ''],
            ['*', '*', 1, '', '', ''],
            ['*', '*', '*', 1, '', ''],
            ['*', '*', '*', '*', 1, ''],
            ['*', '*', '*', '*', '*', 1]
        ], h_buff=0.9).set_color(BLUE).scale(0.65)
        
        # U 矩阵（上三角）- 完全稠密！
        mat_U = Matrix([
            ['*', '*', '*', '*', '*', '*'],
            ['', '*', '*', '*', '*', '*'],
            ['', '', '*', '*', '*', '*'],
            ['', '', '', '*', '*', '*'],
            ['', '', '', '', '*', '*'],
            ['', '', '', '', '', '*']
        ], h_buff=0.9).set_color(GREEN).scale(0.65)
        
        # 布局：A = LU
        decomposition = VGroup(
            mat_A,
            MathTex("="),
            mat_L,
            mat_U
        ).arrange(buff=0.4).shift(UP * .5)
        
        # 标签
        label_A = MathTex("A", font_size=40).next_to(mat_A, UP, buff=0.3)
        label_L = MathTex("L", font_size=40).next_to(mat_L, UP, buff=0.3)
        label_U = MathTex("U", font_size=40).next_to(mat_U, UP, buff=0.3)
        
        # 动画序列
        self.play(Write(mat_A), Write(label_A))
        
        self.wait()
        
        # 显示分解
        self.play(
            Write(decomposition[1]),  # 等号
            Write(mat_L),
            Write(mat_U),
            Write(label_L),
            Write(label_U)
        )
        self.wait()
        
        # 高亮 fill-in 现象 - L 矩阵的所有非对角线元素
        l_entries = mat_L.get_entries()
        # L 的下三角所有 * 元素
        l_fillin_indices = [6, 12, 13, 18, 19, 20, 24, 25, 26, 27, 30, 31, 32, 33, 34]
        l_fillin = VGroup(*[l_entries[i] for i in l_fillin_indices])
        
        # U 矩阵的所有非对角线元素
        u_entries = mat_U.get_entries()
        # U 的上三角所有 * 元素（包括第一行）
        u_fillin_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 14, 15, 16, 17, 21, 22, 23, 28, 29, 35]
        u_fillin = VGroup(*[u_entries[i] for i in u_fillin_indices])

        fill_in = Tex("fill-in", font_size=70).set_color(YELLOW).to_edge(DOWN, buff=1.5)
        
        # 高亮 fill-in 元素
        self.play(
            LaggedStart(*[
                Indicate(entry, color=RED, scale_factor=1.5)
                for entry in l_fillin
            ], lag_ratio=0.03),
            run_time=1.2
        )
        
        self.play(
            LaggedStart(*[
                Indicate(entry, color=RED, scale_factor=1.5)
                for entry in u_fillin
            ], lag_ratio=0.03),
            run_time=1.5
        )
        
        self.wait()

        self.play(Write(fill_in))
        self.wait()


class SpecialLUDecompositions(Scene):
    """展示当矩阵A具有特殊性质时的各种LU分解方法"""
    
    def construct(self):
        # 使用Title类创建标题
        title = Title("特殊的LU分解", font_size=50).shift(UP * .2)
        self.add(title)
        self.wait()
        
        # 创建四种分解方法的展示
        font_size = 32
        
        # 1. LDL分解 (对称矩阵)
        ldl_title = Tex("1. {{LDL分解}}", font_size=font_size).set_color_by_tex("LDL", YELLOW)
        ldl_flops = Tex("{{$\\dfrac{1}{3}n^3$}} flops", font_size=font_size - 8).set_color_by_tex("n", RED)
        ldl_header = VGroup(ldl_title, ldl_flops).arrange(RIGHT, buff=0.3)
        
        ldl_condition = Tex("条件：$A = A^T$", font_size=font_size - 8)
        
        # 统一的scale参数
        ldl_scale = 0.45
        
        mat_A_sym = Matrix([
            [4, 2, 2],
            [2, 3, 1],
            [2, 1, 3]
        ], element_alignment_corner=ORIGIN).scale(ldl_scale)
        
        mat_L_ldl = Matrix([
            [1, '', ''],
            ['*', 1, ''],
            ['*', '*', 1]
        ], element_alignment_corner=ORIGIN).set_color(BLUE).scale(ldl_scale)
        
        mat_D = Matrix([
            ['*', '', ''],
            ['', '*', ''],
            ['', '', '*']
        ], element_alignment_corner=ORIGIN).set_color(PURPLE).scale(ldl_scale)
        
        mat_LT = Matrix([
            [1, '*', '*'],
            ['', 1, '*'],
            ['', '', 1]
        ], element_alignment_corner=ORIGIN).set_color(BLUE).scale(ldl_scale)
        
        ldl_decomp = VGroup(
            mat_A_sym,
            MathTex("=", font_size=35),
            mat_L_ldl,
            mat_D,
            mat_LT
        ).arrange(buff=0.15)
        
        label_A_ldl = MathTex("A", font_size=25).next_to(mat_A_sym, DOWN, buff=0.15)
        label_L_ldl = MathTex("L", font_size=25).next_to(mat_L_ldl, DOWN, buff=0.15)
        label_D_ldl = MathTex("D", font_size=25).next_to(mat_D, DOWN, buff=0.15)
        label_LT_ldl = MathTex("L^T", font_size=25).next_to(mat_LT, DOWN, buff=0.15)
        
        # 让labels底部对齐
        ldl_labels = VGroup(label_A_ldl, label_L_ldl, label_D_ldl, label_LT_ldl)
        for label in ldl_labels:
            label.align_to(ldl_labels, DOWN)
        
        ldl_group = VGroup(
            ldl_header,
            ldl_condition,
            ldl_decomp,
            ldl_labels
        ).arrange(DOWN, buff=0.2, aligned_edge=ORIGIN)
        
        # 2. Cholesky分解 (对称正定矩阵)
        chol_title = Tex("2. {{Cholesky分解}}", font_size=font_size).set_color_by_tex("Cholesky", YELLOW)
        chol_flops = Tex("{{$\\dfrac{1}{3}n^3$}} flops", font_size=font_size - 8).set_color_by_tex("n", RED)
        chol_header = VGroup(chol_title, chol_flops).arrange(RIGHT, buff=0.3)
        
        chol_condition = Tex("条件：$A = A^T$，$A$正定", font_size=font_size - 8)
        
        # 统一的scale参数
        chol_scale = 0.45
        
        mat_A_spd = Matrix([
            [4, 2, 2],
            [2, 3, 1],
            [2, 1, 3]
        ], element_alignment_corner=ORIGIN).scale(chol_scale)
        
        mat_L_chol = Matrix([
            ['*', '', ''],
            ['*', '*', ''],
            ['*', '*', '*']
        ], element_alignment_corner=ORIGIN).set_color(BLUE).scale(chol_scale)
        
        mat_LT_chol = Matrix([
            ['*', '*', '*'],
            ['', '*', '*'],
            ['', '', '*']
        ], element_alignment_corner=ORIGIN).set_color(BLUE).scale(chol_scale)
        
        chol_decomp = VGroup(
            mat_A_spd,
            MathTex("=", font_size=35),
            mat_L_chol,
            mat_LT_chol
        ).arrange(buff=0.15)
        
        label_A_chol = MathTex("A", font_size=25).next_to(mat_A_spd, DOWN, buff=0.15)
        label_L_chol = MathTex("L", font_size=25).next_to(mat_L_chol, DOWN, buff=0.15)
        label_LT_chol = MathTex("L^T", font_size=25).next_to(mat_LT_chol, DOWN, buff=0.15)
        
        # 让labels底部对齐
        chol_labels = VGroup(label_A_chol, label_L_chol, label_LT_chol)
        for label in chol_labels:
            label.align_to(chol_labels, DOWN)
        
        chol_group = VGroup(
            chol_header,
            chol_condition,
            chol_decomp,
            chol_labels
        ).arrange(DOWN, buff=0.2, aligned_edge=ORIGIN)
        
        # 3. 带状矩阵的LU分解
        band_title = Tex("3. {{带状矩阵LU}}", font_size=font_size).set_color_by_tex("带状", YELLOW)
        band_flops = Tex("{{$O(np^2)$}} flops", font_size=font_size - 8).set_color_by_tex("O", RED)
        band_header = VGroup(band_title, band_flops).arrange(RIGHT, buff=0.3)
        
        band_condition = Tex("条件：带宽为$p$", font_size=font_size - 8)
        
        # 统一的scale参数
        band_scale = 0.38
        
        mat_A_band = Matrix([
            ['*', '*', '', '', ''],
            ['*', '*', '*', '', ''],
            ['', '*', '*', '*', ''],
            ['', '', '*', '*', '*'],
            ['', '', '', '*', '*']
        ], h_buff=0.8, element_alignment_corner=ORIGIN).scale(band_scale)
        
        mat_L_band = Matrix([
            [1, '', '', '', ''],
            ['*', 1, '', '', ''],
            ['', '*', 1, '', ''],
            ['', '', '*', 1, ''],
            ['', '', '', '*', 1]
        ], h_buff=0.8, element_alignment_corner=ORIGIN).set_color(BLUE).scale(band_scale)
        
        mat_U_band = Matrix([
            ['*', '*', '', '', ''],
            ['', '*', '*', '', ''],
            ['', '', '*', '*', ''],
            ['', '', '', '*', '*'],
            ['', '', '', '', '*']
        ], h_buff=0.8, element_alignment_corner=ORIGIN).set_color(GREEN).scale(band_scale)
        
        band_decomp = VGroup(
            mat_A_band,
            MathTex("=", font_size=35),
            mat_L_band,
            mat_U_band
        ).arrange(buff=0.15)
        
        label_A_band = MathTex("A", font_size=25).next_to(mat_A_band, DOWN, buff=0.15)
        label_L_band = MathTex("L", font_size=25).next_to(mat_L_band, DOWN, buff=0.15)
        label_U_band = MathTex("U", font_size=25).next_to(mat_U_band, DOWN, buff=0.15)
        
        # 让labels底部对齐
        band_labels = VGroup(label_A_band, label_L_band, label_U_band)
        for label in band_labels:
            label.align_to(band_labels, DOWN)
        
        band_group = VGroup(
            band_header,
            band_condition,
            band_decomp,
            band_labels
        ).arrange(DOWN, buff=0.2, aligned_edge=ORIGIN)
        
        # 4. 块LU分解
        block_title = Tex("4. {{块LU分解}}", font_size=font_size).set_color_by_tex("块LU", YELLOW)
        block_flops = Tex("{{$\\dfrac{2}{3}n^3$}} flops（可并行）", font_size=font_size - 8).set_color_by_tex("n", RED)
        block_header = VGroup(block_title, block_flops).arrange(RIGHT, buff=0.3)
        
        block_condition = Tex("条件：矩阵可分块", font_size=font_size - 8)
        
        # 统一的scale参数
        block_scale = 0.6
        
        # 使用2x2分块矩阵
        mat_A_block = Matrix([
            ["A_{11}", "A_{12}"],
            ["A_{21}", "A_{22}"]
        ], element_alignment_corner=ORIGIN, h_buff=1).scale(block_scale)

        mat_L_block = Matrix([
            ["L_{11}", "0"],
            ["L_{21}", "L_{22}"]
        ], element_alignment_corner=ORIGIN, h_buff=1).set_color(BLUE).scale(block_scale)

        mat_U_block = Matrix([
            ["U_{11}", "U_{12}"],
            ["0", "U_{22}"]
        ], element_alignment_corner=ORIGIN, h_buff=1).set_color(GREEN).scale(block_scale)
        
        block_decomp = VGroup(
            mat_A_block,
            MathTex("=", font_size=35),
            mat_L_block,
            mat_U_block
        ).arrange(buff=0.15)
        
        block_group = VGroup(
            block_header,
            block_condition,
            block_decomp
        ).arrange(DOWN, buff=0.3, aligned_edge=ORIGIN)
        
        # 使用2x2网格布局
        # 左侧：LDL分解和带状矩阵LU分解
        left_group = VGroup(ldl_group, band_group).arrange(DOWN, buff=0.5, aligned_edge=ORIGIN)
        
        # 右侧：Cholesky分解和块LU分解
        right_group = VGroup(chol_group, block_group).arrange(DOWN, buff=0.5, aligned_edge=ORIGIN)
        
        # 左右排列
        all_content = VGroup(left_group, right_group).arrange(RIGHT, buff=1.2)
        all_content.next_to(title, DOWN, buff=.0)
        
        # 底部对齐
        ldl_group.align_to(chol_group, UP)
        band_group.align_to(block_group, UP).shift(DOWN * .1)
        
        # 动画：依次展示四种分解
        self.play(FadeIn(ldl_group, shift=RIGHT))
        self.play(FadeIn(chol_group, shift=LEFT))
        self.play(FadeIn(band_group, shift=RIGHT))
        self.play(FadeIn(block_group, shift=LEFT))
        self.wait(1)

class IterativeMethod(Scene):
    def __init__(self):
        super().__init__()

    def construct(self):
        direct = VGroup(
            Tex("迭代法", font_size=150, stroke_width=4),
            Tex("Iterative Methods", font_size=60, stroke_width=4),
        ).arrange(DOWN).set_color(YELLOW)
        comment = Tex(r"（涉及较深的理论，只做简单介绍，请按需取用\textasciitilde）", font_size=40)
        VGroup(direct, comment).arrange(DOWN, buff=1)

        self.add(comment)
        self.play(Write(direct))
        self.wait()


class IterativeComputationCost(Scene):
    def construct(self):
        # 创建坐标系（不使用对数刻度，手动标注）
        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[-1, 20, 4],
            x_length=8,
            y_length=6,
            axis_config={"include_tip": True},
            x_axis_config={
                "numbers_to_include": [],  # 不显示数字
                "include_ticks": False,  # 不显示刻度
            },
            y_axis_config={
                "numbers_to_include": [],  # 手动添加标签
            },
        ).shift(RIGHT)
        
        # 手动添加 y 轴标签（对数刻度，从上到下）
        y_labels = VGroup()
        y_positions = [0, 4, 8, 12, 16]  # 对应 10^0, 10^-4, 10^-8, 10^-12, 10^-16
        y_exponents = [0, -4, -8, -12, -16]
        for pos, exp in zip(y_positions, y_exponents):
            label = MathTex(f"10^{{{exp}}}", font_size=24)
            label.next_to(axes.c2p(0, 16 - pos), LEFT, buff=0.2)  # 反转：16-pos 使得大数在上
            y_labels.add(label)
        
        # 坐标轴标签 - x轴标签放在底部
        x_label = Tex("计算量", font_size=36).next_to(axes.x_axis, RIGHT, buff=0.25)
        y_label = axes.get_y_axis_label(
            Tex(r"$\Vert b-Ax\Vert$", font_size=36),
            edge=LEFT,
            direction=LEFT,
            buff=1
        )
        
        # 直接法的曲线：先水平（O(1)），然后突然下降到O(eps)
        direct_x1 = 70  # 直接法开始计算的位置
        direct_points = [
            axes.c2p(0, 16),      # 从 10^0 = 1 开始（y=16 对应顶部）
            axes.c2p(direct_x1, 16),  # 保持水平
            axes.c2p(direct_x1, 0),  # 突然下降到 10^-16（机器精度，y=0 对应底部）
            axes.c2p(90, 0),  # 保持在机器精度
        ]
        
        direct_line = VMobject(color=BLUE, stroke_width=4)
        direct_line.set_points_as_corners(direct_points)
        
        # 迭代法的曲线：线性下降，然后保持水平（在直接法的1/3处收敛）
        iter_x1 = direct_x1 / 3  # 迭代法达到收敛的位置（约23）
        iter_points = [
            axes.c2p(0, 16),      # 从 10^0 = 1 开始（y=16 对应顶部）
            axes.c2p(iter_x1, 0),  # 线性下降到 10^-16（机器精度，y=0 对应底部）
            axes.c2p(40, 0),  # 保持在收敛精度
        ]
        
        iter_line = VMobject(color=YELLOW, stroke_width=4)
        iter_line.set_points_as_corners(iter_points)
        
        # 图例
        direct_legend = Tex("direct", font_size=32, color=BLUE).next_to(axes.c2p(direct_x1 / 2, 16), UP)
        
        iter_legend = Tex("iterative", font_size=32, color=YELLOW).next_to(axes.c2p(direct_x1 / 6, 8), RIGHT)
        
        # 标注关键点
        iter_converge = Dot(axes.c2p(iter_x1, 0), color=YELLOW, radius=0.08)
        
        direct_annotation = MathTex(r"O(n^3)", font_size=28, color=BLUE).next_to(
            axes.c2p(direct_x1, 0), DOWN, buff=0.2
        )
        iter_annotation = Tex(r"收敛", font_size=28, color=YELLOW).next_to(
            iter_converge, DOWN, buff=0.2
        )
        
        # 动画序列
        self.add(axes)
        self.play(Write(x_label), Write(y_label), Write(y_labels))
        self.wait()
        
        self.play(Create(direct_line), run_time=2)
        self.play(Write(direct_legend), Write(direct_annotation))
        self.wait()

        self.wait()
        self.play(Create(iter_line), run_time=2)
        self.play(Write(iter_legend))
        self.wait()
        self.play(FadeIn(iter_converge), Write(iter_annotation))
        self.wait()


class BigDefinition(Scene):
    """展示不同年代"大矩阵"的定义演变"""
    
    def construct(self):
        # 标题
        self.title = Title("什么算``大''矩阵？", font_size=48)
        self.add(self.title)
        self.wait()
        
        # 历史数据
        timeline_data = [
            (1950, 20, "真空管计算机时代"),
            (1965, 200, "晶体管计算机"),
            (1980, 2000, "集成电路时代"),
            (1995, 20000, "个人计算机普及"),
            (2010, 100000, "多核处理器"),
            (2020, 500000, "GPU加速计算"),
            (2026, 2000000, "AI芯片 + 分布式")
        ]
        
        # 创建时间轴
        self.show_timeline(timeline_data)

    def show_timeline(self, data):
        """显示时间轴动画"""
        timeline_group = VGroup()
        
        # 计算布局
        start_y = 2.5
        spacing = 0.85
        
        for i, (year, size, desc) in enumerate(data):
            y_pos = start_y - i * spacing
            
            # 年份
            year_text = Text(f"{year}:", font_size=32, color=BLUE)
            year_text.move_to(LEFT * 5 + UP * y_pos)
            
            # 矩阵规模 (使用 n >= 格式)
            size_formula = MathTex(f"n \\geq {size:,}", font_size=40, color=YELLOW)
            size_formula.next_to(year_text, RIGHT, buff=0.5)
            
            # 描述
            desc_text = Text(f"({desc})", font_size=22, color=GRAY)
            desc_text.next_to(size_formula, RIGHT, buff=0.4)
            
            row_group = VGroup(year_text, size_formula, desc_text)
            timeline_group.add(row_group)

        self.timeline_group = timeline_group.next_to(self.title, DOWN, buff=.5)
        self.play(FadeIn(self.timeline_group))
        self.wait()
    
class KrylovSubspace(Scene):
    def construct(self):
        # 标题
        title = Title("Krylov Subspace")
        self.add(title)
        self.wait()
        
        # 具体例子：2x2矩阵
        example_matrix = MathTex(
            r"A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}", r",~ p(\lambda) = \det(\lambda I - A) = {{\lambda}}^2 - 6{{\lambda}} + 8 = 0",
            font_size=44
        ).set_color_by_tex_to_color_map({'A': RED, 'lambda': BLUE})
        example_matrix[1].set_color(WHITE)
        for i in (3, 10):
            example_matrix[1][i].set_color(BLUE)
        example_matrix[1][13].set_color(RED)
        # Cayley-Hamilton定理
        cayley_hamilton = MathTex(
            r"\Longrightarrow p({{A}}) = {{A}}^2 - 6{{A}} + 8I = O",
            font_size=44,
        ).set_color_by_tex_to_color_map({'A': RED})
        
        inverse_result = MathTex(
            r"\Longrightarrow {{A}}^{-1} = -\frac{1}{8}{{A}} + \frac{3}{4}I",
            font_size=44,
        )

        VGroup(
            example_matrix,
            cayley_hamilton,
            inverse_result
        ).arrange(DOWN).next_to(title, DOWN)
        # 逆矩阵的多项式表示
        inverse_poly = MathTex(
            r"{{x=A^{-1}b = p_{n-1}(A)b}} \qquad p_{n-1} \in \mathcal{P}_{n-1}",
            font_size=44,
        ).set_color_by_tex('A', YELLOW).next_to(inverse_result, DOWN, buff=0.5)
        box = SurroundingRectangle(inverse_poly[0], color=YELLOW, buff=0.2)
        
        
        self.play(Write(example_matrix))
        self.wait()
        self.play(Write(cayley_hamilton))
        self.wait()
        self.play(Write(inverse_result))
        self.wait()
        self.play(Create(box), Write(inverse_poly))
        self.wait()
        self.play(
            FadeOut(VGroup(example_matrix, cayley_hamilton, inverse_result)),
            VGroup(box, inverse_poly).animate.next_to(title, DOWN, buff=.5)
        )
        self.wait()
        chain = MathTex(
            r"p_0(A)b~\subseteq~ p_1(A)b~\subseteq~ p_2(A)b~\subseteq~\cdots,\quad p_k\in \mathcal{P}_k",
            color=BLUE,
            font_size=35
        ).next_to(inverse_poly, DOWN, buff=1)
        self.play(Write(chain))
        self.wait()
        # Krylov子空间定义
        krylov_def = MathTex(
            r"\mathcal{K}_k", r"= \{", "p_{k-1}(A)b", r"\}", r"=", r"\mathrm{span}\{b, Ab, A^2b, \ldots, A^{k-1}b\}",
            font_size=44,
        ).set_color_by_tex_to_color_map(
            {'(A)b': BLUE, 'span': YELLOW}
        ).next_to(chain, DOWN, buff=0.8)
        
        self.play(Write(krylov_def[:4]))
        self.wait()
        self.play(Write(krylov_def[4:]))
        self.wait()

        basis = Tex(r"一组（很差的）基：", "$b, ~Ab, ~A^2b, ~\ldots, ~A^{k-1}b$").next_to(krylov_def, DOWN, buff=1)
        basis[1].set_color(YELLOW)
        self.play(Write(basis))
        self.wait()
        # self.add(index_labels(krylov_def))

class PowerMethod(LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False,
            show_coordinates=False,
            leave_ghost_vectors=False,
            include_foreground_plane=True,
        )
    
    def construct(self):
        # 定义矩阵 A
        matrix = [[1.5, 0.5], [0.5, 1.5]]
        
        # 初始向量 b
        initial_vector = np.array([0, 1])
        initial_vector = initial_vector# / np.linalg.norm(initial_vector)
        
        # 创建初始向量 b
        vector_b = Vector(initial_vector, color=YELLOW)
        label_b = MathTex("b", color=YELLOW).move_to(
            np.array([*initial_vector, 0]) + np.array([-0.3, 0.2, 0])
        )
        self.add_foreground_mobject(label_b)
        self.add(vector_b, label_b)
        self.wait()
        
        # 迭代生成 Ab, A^2b, A^3b, ...
        current_vector = initial_vector.copy()
        vectors = [vector_b]
        labels = [label_b]
        
        iterations = 8
        # 生成从黄色到绿色的渐变，包括 b 在内共 7 个向量
        colors = [interpolate_color(YELLOW, GREEN, alpha) for alpha in np.linspace(0, 1, iterations + 1)]
        # 第一个颜色已经用于 b，所以从第二个开始
        colors = colors[1:]
        
        for i in range(iterations):
            # 应用矩阵变换
            next_vector = np.dot(matrix, current_vector)
            next_vector = next_vector / 1.5
            
            # 创建新向量
            new_vector = Vector(next_vector, color=colors[i])
            vector_end = np.array([*next_vector, 0])# * 1.3
            new_label = MathTex(f"A^{{{i+1}}}b", color=colors[i]).move_to(
                vector_end + np.array([0.3, 0.2, 0])
            )
            
            vectors.append(new_vector)
            labels.append(new_label)
            
            # self.add_foreground_mobject(new_label)
            self.add(new_vector)
            self.wait(.2)
            
            current_vector = next_vector
        
        self.wait()


class Arnoldi(Scene):
    def construct(self):
        title = Title("Krylov子空间的一组正交基")
        self.add(title)
        self.wait()
        cm = {'h': BLUE, 'q': RED}
        algo = VGroup(
            MathTex(r"{{q_1}} = b/\lVert b\rVert").set_color_by_tex_to_color_map(cm),
            Tex(r"\textbf{for} $k=1,2,3,\ldots$"),
            MathTex(r"v = A{{q_k}}").set_color_by_tex_to_color_map(cm),
            Tex(r"\textbf{for} $j=1$ \textbf{to} $k$"),
            MathTex(r"{{h_{jk}}}=", "q_j^T", "v").set_color_by_tex_to_color_map(cm),
            MathTex(r"v=v-{{h_{j k}}}{{q_j}}").set_color_by_tex_to_color_map(cm),
            MathTex(r"{{h_{k+1,k}}}=\lVert v\rVert").set_color_by_tex_to_color_map(cm),
            MathTex(r"{{q_{k+1}}} = v / {{h_{k+1,k}}}").set_color_by_tex_to_color_map(cm)
        ).arrange(DOWN, aligned_edge=LEFT)
        algo[2:].shift(RIGHT)
        algo[4:6].shift(RIGHT)
        algo.scale(.8).next_to(title, DOWN, buff=1).to_edge(LEFT, buff=1)
        box = SurroundingRectangle(algo).stretch(.7, dim=0)
        name = Tex("Arnoldi Iteration", color=box.get_color(), font_size=35).next_to(box, UP)
        self.play(Write(algo))
        self.play(Create(box), Write(name))
        # self.add(index_labels(algo[4]))

        # Arnoldi迭代结果展示：AQ_k = Q_{k+1}H_k
        k = 4  # k=4
        n = 8  # 矩阵维度
        opacity = 1  # 填充颜色的不透明度
        
        # 创建矩阵A的grid (n x n)
        a_grid = VGroup(
            *[
                VGroup(*[Square(0.3, stroke_width=0).set_fill(WHITE, opacity=opacity) for _ in range(n)]).arrange(RIGHT, buff=.05)
                for _ in range(n)
            ]
        ).arrange(DOWN, buff=.05)
        
        # 创建矩阵Q_k的grid (n x k)，每列用不同深浅的红色
        red_colors = [RED_E, RED_D, RED_C, RED_B]  # 四种深浅的红色
        q_grid = VGroup(
            *[
                VGroup(*[Square(0.3, stroke_width=0).set_fill(red_colors[j], opacity=opacity) 
                        for j in range(k)]).arrange(RIGHT, buff=.05)
                for _ in range(n)
            ]
        ).arrange(DOWN, buff=.05)
        
        # 等号
        equals1 = MathTex("=", font_size=50)
        
        # 创建矩阵Q_{k+1}的grid (n x (k+1))
        red_colors_extended = [RED_E, RED_D, RED_C, RED_B, RED_A]  # 五种深浅的红色
        q_plus_grid = VGroup(
            *[
                VGroup(*[Square(0.3, stroke_width=0).set_fill(red_colors_extended[j], opacity=opacity) 
                        for j in range(k+1)]).arrange(RIGHT, buff=.05)
                for _ in range(n)
            ]
        ).arrange(DOWN, buff=.05)
        
        # 创建上Hessenberg矩阵H_k的grid ((k+1) x k)
        # 只在上Hessenberg部分（主对角线+上三角+第一条次对角线）显示方块
        h_grid = VGroup()
        for i in range(k+1):
            row = VGroup()
            for j in range(k):
                if j >= i - 1:  # 上Hessenberg条件：j >= i-1
                    square = Square(0.3, stroke_width=0).set_fill(BLUE, opacity=opacity)
                else:
                    square = Square(0.3, fill_opacity=0, stroke_opacity=0)  # 不显示
                row.add(square)
            row.arrange(RIGHT, buff=.05)
            h_grid.add(row)
        h_grid.arrange(DOWN, buff=.05)
        
        # 排列所有元素：A Q_k = Q_{k+1} H_k
        equation = VGroup(
            a_grid,
            q_grid,
            equals1,
            q_plus_grid,
            h_grid
        ).arrange(RIGHT, buff=0.4).scale(0.8).to_edge(RIGHT, buff=1).shift(UP * .2)
        a_label = MathTex("A", font_size=40).next_to(a_grid, DOWN)
        q_label = MathTex("Q_k", font_size=40, color=RED).next_to(q_grid, DOWN)
        q_plus_label = MathTex("Q_{k+1}", font_size=40, color=RED).next_to(q_plus_grid, DOWN)
        h_label = MathTex("\\tilde H_k", font_size=40, color=BLUE).next_to(h_grid, DOWN).align_to(q_plus_label, DOWN)
        
        hessenberg = Tex("$\\tilde H_k$: upper Hessenberg", color=BLUE).next_to(equation, DOWN, buff=1.2)

        # 动画展示
        self.play(FadeIn(equation), FadeIn(a_label), FadeIn(q_label), FadeIn(q_plus_label), FadeIn(h_label))
        self.wait()

        self.play(Write(hessenberg))
        self.wait()

class GMRES(Scene):
    def construct(self):
        cm = {'Q': RED, 'H': BLUE}
        title = Title("GMRES (Generalized Minimal Residual Method)")
        self.add(title)
        self.wait()

        idea = MathTex(r"\min_{x_k\in \mathcal{K}_k} \lVert b-Ax_k\rVert", font_size=60)
        idea.next_to(title, DOWN, buff=.5)

        inter = MathTex(r"A", r"Q_k", "=", "Q_{k+1}", r"\tilde H_k", r",~", "x_k", "=", "Q_k", "y").set_color_by_tex_to_color_map(cm)

        ls = MathTex(r"\Longrightarrow \lVert r_k\rVert= ", r"\lVert \tilde H_k y -\lVert b\rVert e_1\rVert").set_color_by_tex('H', YELLOW)
        VGroup(inter, ls).arrange().next_to(idea, DOWN)

        algo = VGroup(
            Tex(r"\textbf{for} $k=1,2,3,\ldots$"),
            Tex(r"\textit{Step $k$ of Arnoldi...}", color=GREEN),
            MathTex(r"\min_y ~\lVert \tilde H_k y -\lVert b\rVert e_1\rVert"),
            MathTex("x_n=Q_n y")
        ).arrange(DOWN, aligned_edge=LEFT).scale(.7)
        algo[1:].shift(RIGHT * .7)
        algo.to_corner(DL, buff=1)
        box = SurroundingRectangle(algo, buff=.2)
        algo_name = Tex("GMRES", color=YELLOW, font_size=35).next_to(box, UP)

        arrow = Arrow(ORIGIN, LEFT, color=GREEN).set_stroke(width=5).next_to(algo[-2], RIGHT, buff=.2)
        cost = MathTex("O(k^2)", color=GREEN, font_size=35).next_to(arrow)

        convergence = VGroup(
            Tex("若$A$可对角化：$A=X\Lambda X^{-1}$，那么", font_size=35, color=BLUE),
            MathTex(r"\lVert r_k\rVert \leq \kappa(X)\min_{p\in\mathcal{P}_k,~p(0)=1}\max_{z\in \Lambda}|p(z)|", color=YELLOW)
        ).arrange(DOWN, buff=.5).to_corner(DR).align_to(algo_name, UP)

        comment = Tex(r"\kaishu 收敛很快：当$A$的特征值在\\远离原点的地方集中分布", font_size=40, color=BLUE).next_to(convergence, DOWN, buff=.5)

        self.play(Write(idea))
        self.wait()

        self.play(Write(inter))
        self.wait()

        self.play(Write(ls))
        self.wait()

        self.play(Write(algo))
        self.play(Create(box), Write(algo_name))
        self.wait()

        self.play(GrowArrow(arrow), Write(cost))
        self.wait()
        self.play(FadeOut(arrow), FadeOut(cost))
        self.play(Write(convergence))
        self.wait()
        self.play(Write(comment))
        self.wait()

class Residual(LinearTransformationScene):
    def __init__(self, **kwargs):
        super().__init__(
            leave_ghost_vectors=True,
            show_basis_vectors=False,
            **kwargs
        )
        self.A = [[2, 2], [1, 3]]
        self.b = (-4, -1)
        self.x_star = np.linalg.solve(self.A, self.b)  # 真实解
        self.x_k = np.array([-1, 1])  # 当前迭代解
        
    def construct(self):
        
        # 先画出 x^* 和 x_k
        x_star_vec = self.add_vector(self.x_star, color=YELLOW)
        x_star_label = (
            MathTex(r"x^*", color=x_star_vec.get_color())
            # .add_background_rectangle()
            .next_to(x_star_vec.get_end(), LEFT)
        )
        
        x_k_vec = self.add_vector(self.x_k, color=GOLD)
        x_k_label = (
            MathTex(r"x_k", color=x_k_vec.get_color())
            # .add_background_rectangle()
            .next_to(x_k_vec.get_end(), UP)
        )
        
        self.play(Write(x_star_label), Write(x_k_label))
        self.wait()
        
        # 画出 e_k = x^* - x_k（作为箭头，从 x_k 指向 x_star）
        e_k_arrow = Arrow(
            start=x_k_vec.get_end(),
            end=x_star_vec.get_end(),
            color=RED,
            buff=0
        )
        
        e_k_label = (
            MathTex(r"e_k = x^* - x_k", color=e_k_arrow.get_color())
            .next_to(e_k_arrow.get_center(), UL)
        )
        
        self.play(GrowArrow(e_k_arrow), FadeIn(e_k_label))
        self.wait()
        
        # 应用 A: x^* -> b, x_k -> Ax_k, e_k -> r_k
        # 创建新标签（变换后的）
        b_label = MathTex(r"b", color=YELLOW)
        ax_k_label = MathTex(r"Ax_k", color=GOLD)
        r_k_label = MathTex(r"r_k = b - Ax_k", color=RED)
        
        # 手动对每个向量应用矩阵变换
        # 计算变换后的位置
        A_np = np.array(self.A)
        b = A_np @ self.x_star
        Ax_k = A_np @ self.x_k
        
        # 先将新标签移到目标位置（用于计算位置）
        b_label.next_to(self.plane.c2p(*b), DOWN)
        ax_k_label.next_to(self.plane.c2p(*Ax_k), UP)
        r_k_label.next_to(
            (self.plane.c2p(*Ax_k) + self.plane.c2p(*b)) / 2,
            UL
        )
        
        # 创建变换动画
        self.play(
            # 变换网格
            self.background_plane.animate.apply_matrix(A_np),
            self.plane.animate.apply_matrix(A_np),
            # 变换向量
            x_star_vec.animate.put_start_and_end_on(ORIGIN, self.plane.c2p(*b)),
            x_k_vec.animate.put_start_and_end_on(ORIGIN, self.plane.c2p(*Ax_k)),
            # 变换箭头
            e_k_arrow.animate.put_start_and_end_on(
                self.plane.c2p(*Ax_k),
                self.plane.c2p(*b)
            ),
            # 变换标签
            ReplacementTransform(x_star_label, b_label),
            ReplacementTransform(x_k_label, ax_k_label),
            ReplacementTransform(e_k_label, r_k_label),
            run_time=2
        )
        
        self.wait()


class CG1(ThreeDScene):
    def __init__(self):
        super().__init__()
        self.A = np.array([
            [3, 2],
            [2, 6]
        ], dtype=float)
        self.b = np.array([2, -8], dtype=float)
        
        # 计算函数 f(x) = 1/2 x^T A x - b^T x
        def f(x, y):
            vec = np.array([x, y])
            return 0.5 * vec @ self.A @ vec - self.b @ vec
        
        self.f = f
    
    def construct(self):
        # 标题：共轭梯度法
        title = Title("共轭梯度法（Conjugate Gradient Method, CG）").set_color(YELLOW)
        
        # 第一行公式：f(x) = 1/2 x^T A x - b^T x
        formula1 = MathTex(
            r"\text{minimize } f(x) = {{\frac{1}{2} x^T A x}} {{- b^T x}}",
            font_size=50
        ).set_color_by_tex_to_color_map({'A': YELLOW, 'b': BLUE})
        
        # 双箭头（等价符号）
        iff_arrow = MathTex(
            r"\Longleftrightarrow", font_size=50
        ).rotate(DEGREES * 90)
        
        # 第二行公式：∇f(x) = Ax - b = 0，以及 ∇²f(x) ≻ 0
        formula2 = MathTex(
            r"\text{solve } \nabla f(x) = {{Ax}} {{- b}} = 0",
            font_size=50
        ).set_color_by_tex_to_color_map({'A': YELLOW, '- b': BLUE})
        posdef = MathTex(
            r"\nabla^2f(x)={{A}}\succ 0",
            font_size=30
        ).to_corner(DL, buff=2).set_color_by_tex('A', YELLOW)
        # 排列公式
        formulas = VGroup(formula1, iff_arrow, formula2).arrange(DOWN, buff=0.3).move_to([2, -.3, 0])
        # 创建3D坐标系和曲面
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-12, 6, 1],
            x_length=4,
            y_length=6,
            z_length=2,
        ).move_to([-10, 3, -2])
        
        surface = Surface(
            lambda u, v: axes.c2p(u, v, self.f(u, v)),
            u_range=[0, 4],
            v_range=[-4, 0],
            resolution=32,
            fill_opacity=0.8,
        )
        
        # 根据z值设置颜色渐变
        surface.set_fill_by_value(axes=axes, colorscale=[(GREEN, -9), (RED, 2)], axis=2)
        
        # 显示标题和公式
        self.add_fixed_in_frame_mobjects(title)
        self.wait()
        self.camera.add_fixed_in_frame_mobjects(formulas, posdef)

        # 计算最小值点：x* = A^{-1}b
        x_star = np.linalg.solve(self.A, self.b)  # 应该是 [2, -2]
        # 最小值点的3D坐标（在axes坐标系中）
        min_point_3d = axes.c2p(x_star[0], x_star[1], self.f(x_star[0], x_star[1]))
        
        # 设置相机初始位置
        self.set_camera_orientation(
            phi=70 * DEGREES,
            theta=-45 * DEGREES,
        )
        
        # 显示坐标系和曲面
        self.play(Write(formula1), Create(surface))
        self.wait(5)
        # 旋转速度：每秒旋转15度（PI/12弧度）
        rotation_speed = PI / 12
        
        def rotate_updater(mob, dt):
            mob.rotate(
                angle=rotation_speed * dt,
                axis=OUT,
                about_point=min_point_3d
            )
        
        # 将surface和球组合在一起旋转
        surface.add_updater(rotate_updater)
        
        # 在写formula2的同时，标注最小值点
        min_point_sphere = Sphere(
            center=min_point_3d,
            radius=0.15,
        ).set_color(YELLOW)
        
        # 创建一个向上的箭头指向最小值点（箭头在点的下方）
        # 使用Z轴方向（[0, 0, -1]表示向下，[0, 0, 1]表示向上）
        arrow_start = min_point_3d + np.array([0, 0, -2])  # Z轴向下2个单位
        arrow_end = min_point_3d + np.array([0, 0, -0.3])  # Z轴向下0.3个单位
        arrow = Arrow3D(
            start=arrow_start,
            end=arrow_end,
            color=YELLOW,
            thickness=0.02,
            height=0.3,
            base_radius=0.15
        )

        # 同时显示formula2和最小值点标注
        self.play(Write(iff_arrow), Write(formula2))
        self.add(min_point_sphere, arrow)
        self.wait(5)
        self.play(Write(posdef))
        self.wait(10)


class CG2(MovingCameraScene):
    def __init__(self):
        super().__init__()
        self.A = np.array([
            [3, 2],
            [2, 6]
        ], dtype=float)
        self.b = np.array([2, -8], dtype=float)
        self.x0 = np.array([-2, -2], dtype=float)
        
        # 计算函数 f(x) = 1/2 x^T A x - b^T x
        def f(x, y):
            vec = np.array([x, y])
            return 0.5 * vec @ self.A @ vec - self.b @ vec
        
        # 计算梯度 grad f(x) = A x - b
        def grad_f(x):
            return self.A @ x - self.b
        
        self.f = f
        self.grad_f = grad_f

    def construct(self):
        # 创建铺满屏幕的网格
        plane = Axes(
            x_range=[-7, 7, 1],
            y_range=[-4, 4, 1],
            x_length=14,
            y_length=8,
            tips=False,
        )
        
        # 创建等值线公式（第一行）
        formula = MathTex(
            r"f(x) = {{\frac{1}{2} x^T A x}} {{- b^T x}} = {{\text{constant}}}",
            font_size=36
        ).set_color_by_tex_to_color_map({'A': YELLOW, 'b': BLUE}).add_background_rectangle()

        formula[6].set_color_by_gradient(RED, GREEN)
        
        # 创建展开的式子（第二行）
        # A = [[3, 2], [2, 6]], b = [2, -8]
        expanded = MathTex(
            r"{{\frac{3}{2} x_1^2 + 2 x_1 x_2 + 3 x_2^2}} {{- 2 x_1 + 8 x_2}} = {{\text{constant}}}",
            font_size=36
        ).set_color_by_tex_to_color_map({'c': RED, 'x_1 x_2': YELLOW, '8 x_2': BLUE}).add_background_rectangle()
        expanded[5].set_color_by_gradient(RED, GREEN)

        VGroup(formula, expanded).arrange(DOWN).to_corner(UL)

        self.play(FadeIn(plane), Write(formula))
        self.wait()
        self.play(
            ReplacementTransform(formula[0].copy(), expanded[0]),
            ReplacementTransform(formula[1:].copy(), expanded[1:]),
        )
        # self.play(Write(expanded))
        self.add_foreground_mobjects(formula, expanded)
        self.wait()
        
        # 计算函数值的范围用于确定等值线的层级
        x_vals = np.linspace(-7, 7, 50)
        y_vals = np.linspace(-4, 4, 50)
        z_vals = []
        for x in x_vals:
            for y in y_vals:
                z_vals.append(self.f(x, y))
        z_min, z_max = min(z_vals), max(z_vals)
        
        # 绘制等值线（使用 ImplicitFunction）
        # 使用非线性分布：在靠近最小值的地方更密集
        # 使用平方根函数来生成更密集的小值等值线
        num_contours = 15
        t_vals = np.linspace(0, 1, num_contours)
        # 使用 t^2 映射，使得小的 t 值更密集
        contour_levels = z_min + (z_max - z_min) * (t_vals ** 2)
        contours = VGroup()
        contour_labels = VGroup()
        
        for i, level in enumerate(contour_levels):
            # 根据函数值设置颜色（红色表示大值，绿色表示小值）
            t = (level - z_min) / (z_max - z_min) if z_max > z_min else 0
            color = interpolate_color(GREEN, RED, t)
            
            # 创建隐函数：f(x, y) - level = 0
            contour = ImplicitFunction(
                lambda x, y: self.f(x, y) - level,
                x_range=[-7, 7],
                y_range=[-4, 4],
                color=color,
                stroke_width=2,
                min_depth=7 if i == 0 else 5,
            )
            contours.add(contour)
        
        
        # 在等值线上添加数值标签
        for i, level in enumerate(contour_levels):
            t = (level - z_min) / (z_max - z_min) if z_max > z_min else 0
            color = interpolate_color(GREEN, RED, t)
            
            # 在等值线的右侧添加标签
            label = MathTex(f"{level:.1f}", font_size=24, color=color)
            label.add_background_rectangle(opacity=0.8)
            
            # 找一个合适的位置放置标签
            x_pos = 2.8
            # 通过求解等值线方程找到对应的y值
            # f(x, y) = level
            # 展开：3/2 * x^2 + 2 * x * y + 3 * y^2 - 2 * x + 8 * y = level
            # 整理成关于y的二次方程：3 * y^2 + (2*x + 8) * y + (3/2 * x^2 - 2*x - level) = 0
            a = 3
            b = 2 * x_pos + 8
            c_coef = 1.5 * x_pos**2 - 2 * x_pos - level
            discriminant = b**2 - 4*a*c_coef
            if discriminant >= 0:
                y_pos = (-b + np.sqrt(discriminant)) / (2*a)
                label.move_to(plane.c2p(x_pos, y_pos, 0))
                contour_labels.add(label)
        
        self.play(
            FadeIn(contours),
            FadeIn(contour_labels), 
            run_time=1
        )
        self.wait()
        
        # 执行最速下降算法
        x_current = self.x0.copy()
        trajectory = VGroup()
        points_group = VGroup()
        
        # 起始点
        start_dot = Dot(
            plane.c2p(x_current[0], x_current[1], 0),
            color=YELLOW,
            radius=0.08
        )
        points_group.add(start_dot)
        self.play(FadeIn(start_dot, scale=3))
        
        # 迭代次数
        num_iterations = 8
        
        for i in range(num_iterations):
            # 计算梯度
            grad = self.grad_f(x_current)
            
            # 计算步长（精确线搜索）
            # alpha = (grad^T grad) / (grad^T A grad)
            alpha = (grad @ grad) / (grad @ self.A @ grad)
            
            # 更新位置
            x_next = x_current - alpha * grad
            
            # 绘制轨迹线
            line = Line(
                plane.c2p(x_current[0], x_current[1], 0),
                plane.c2p(x_next[0], x_next[1], 0),
                color=YELLOW,
                stroke_width=4
            )
            trajectory.add(line)
            
            # 绘制新点
            next_dot = Dot(
                plane.c2p(x_next[0], x_next[1], 0),
                color=YELLOW,
                radius=0.08
            )
            points_group.add(next_dot)
            
            self.play(
                Create(line),
                FadeIn(next_dot),
            )
            
            x_current = x_next
            
            # 如果梯度很小，提前停止
            if np.linalg.norm(grad) < 1e-3:
                break
        
        self.wait()
        
        # 将最速下降的轨迹和点变成蓝色并变淡
        self.play(
            trajectory.animate.set_color(BLUE).set_opacity(0.4),
            points_group.animate.set_color(BLUE).set_opacity(0.4),
        )
        self.wait()
        
        # 执行共轭梯度（CG）算法
        x_current = self.x0.copy()
        cg_trajectory = VGroup()
        cg_points_group = VGroup()
        
        # 起始点（重新标记为黄色）
        cg_start_dot = Dot(
            plane.c2p(x_current[0], x_current[1], 0),
            color=YELLOW,
            radius=0.08
        )
        cg_points_group.add(cg_start_dot)
        self.play(FadeIn(cg_start_dot, scale=3))
        
        # CG算法初始化
        r = self.grad_f(x_current)  # 初始残差（即梯度）
        p = -r.copy()  # 初始搜索方向
        
        # CG迭代（执行2步）
        num_cg_iterations = 2
        
        for i in range(num_cg_iterations):
            # 计算步长
            # alpha = (r^T r) / (p^T A p)
            Ap = self.A @ p
            alpha = (r @ r) / (p @ Ap)
            
            # 更新位置
            x_next = x_current + alpha * p
            
            # 绘制轨迹线
            cg_line = Line(
                plane.c2p(x_current[0], x_current[1], 0),
                plane.c2p(x_next[0], x_next[1], 0),
                color=YELLOW,
                stroke_width=4
            )
            cg_trajectory.add(cg_line)
            
            # 绘制新点
            cg_next_dot = Dot(
                plane.c2p(x_next[0], x_next[1], 0),
                color=YELLOW,
                radius=0.08
            )
            cg_points_group.add(cg_next_dot)
            
            self.play(
                Create(cg_line),
                FadeIn(cg_next_dot),
            )
            self.wait()

            # 更新残差
            r_next = r + alpha * Ap
            
            # 计算 beta（共轭方向的系数）
            beta = (r_next @ r_next) / (r @ r)
            
            # 更新搜索方向
            p = -r_next + beta * p
            
            # 更新当前位置和残差
            x_current = x_next
            r = r_next
            
            # 如果残差很小，提前停止
            if np.linalg.norm(r) < 1e-3:
                break
        
        self.wait()
        p0 = MathTex("p_0", color=YELLOW).move_to(cg_trajectory[0]).shift(UL * .2)
        p1 = MathTex("p_1", color=YELLOW).move_to(cg_trajectory[1]).shift(UR * .2)
        grad = MathTex(r"-\nabla f(x_0)", font_size=30).add_background_rectangle().move_to(cg_trajectory[0]).shift(DR * .4)
        
        # 完全淡出蓝色的最速下降轨迹
        self.play(
            FadeOut(trajectory),
            FadeOut(points_group),
            FadeOut(contour_labels),
            FadeOut(formula),
            FadeOut(expanded)
        )
        self.wait()
        
        # 使用 Cholesky 分解来变换椭圆为圆
        # A = R^T R (Cholesky 分解)
        # 我们需要用 R^{-T} 来变换，使得椭圆变成圆
        R = np.linalg.cholesky(self.A, upper=True)  # 上三角 Cholesky 因子
        
        # 创建 3x3 变换矩阵（用于 Manim 的 3D 坐标）
        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] = R 

        # 应用变换到所有对象，并同时缩小视图
        all_objects = VGroup(plane, contours, cg_trajectory, cg_points_group)
        self.play(
            all_objects.animate.apply_matrix(transform_matrix),
            self.camera.frame.animate.scale(1.5),
            run_time=2
        )

        # 添加直角符号表示两条路径垂直
        if len(cg_trajectory) >= 2:
            # 获取两条路径的交点（第二个点的位置）
            intersection_point = cg_points_group[1].get_center()
            
            # 获取两条路径的方向向量
            line1 = cg_trajectory[0]
            line2 = cg_trajectory[1]
            
            # 计算方向向量（修正方向）
            dir1 = line1.get_end() - line1.get_start()
            dir1 = dir1 / np.linalg.norm(dir1)  # 归一化
            
            dir2 = line2.get_end() - line2.get_start()
            dir2 = dir2 / np.linalg.norm(dir2)  # 归一化
            
            # 创建直角符号（小正方形）- 修正顺序
            right_angle_size = 0.3
            corner1 = intersection_point + right_angle_size * dir2
            corner2 = corner1 - right_angle_size * dir1
            corner3 = intersection_point - right_angle_size * dir1
            
            right_angle = VMobject()
            right_angle.set_points_as_corners([
                corner1,
                corner2,
                corner3
            ])
            right_angle.set_stroke(color=YELLOW, width=4)
            
            # 显示直角符号
            self.play(Create(right_angle))
        
        self.wait()
        # 计算逆矩阵
        inverse_matrix = np.linalg.inv(transform_matrix)
        all_objects.add(right_angle)
        self.play(
            # FadeOut(right_angle),
            all_objects.animate.apply_matrix(inverse_matrix),
            self.camera.frame.animate.scale(1/1.5),
        )
        self.play(
            FadeIn(p0),
            FadeIn(p1)
        )
        self.play(Write(grad))
        self.wait()


class CG3(Scene):
    def __init__(self):
        super().__init__()
        font_size = 40
        self.i_conj = Tex(r"关于$I$正交： $\langle x,y\rangle=x^Ty=0$", font_size=font_size)
        self.a_conj = Tex(r"关于$A$正交：$\langle x,y\rangle_A=x^TAy=0$ ", font_size=font_size)
        self.aside = Tex("($A$-conjugate)", font_size=30)
        self.aside2 = Tex("(需要$A$正定)", font_size=30)
        self.a_norm = MathTex(r"A\text{-norm: } \lVert x\rVert_A=\sqrt{\langle x,x\rangle_A}=\sqrt{x^T A x}", font_size=font_size)

        for i in (7, 12):
            self.i_conj[0][i].set_color(RED)
        for i in (9, 14):
            self.i_conj[0][i].set_color(GREEN)
        for i in (11, 15):
            self.a_conj[0][i].set_color(YELLOW)
        for i in (7, 13):
            self.a_conj[0][i].set_color(RED)
        for i in (9, 16):
            self.a_conj[0][i].set_color(GREEN)
        
        for i in (8, 15, 17, 23, 26):
            self.a_norm[0][i].set_color(RED)
        for i in (10, 19, 25):
            self.a_norm[0][i].set_color(YELLOW)

        VGroup(self.i_conj, self.a_conj, self.a_norm).arrange(DOWN).to_edge(UP)
        self.aside.next_to(self.a_conj, buff=.5)
        self.aside2.next_to(self.a_norm, buff=.5)

        self.algo = VGroup(
            MathTex("x_0=0,~r_0=b,~p_0=r_0"),
            Tex(r"\textbf{for} $k=1,2,3,\ldots$"),
            MathTex(r"\alpha_k = (r_{k-1}^T r_{k-1}) / (p_{k-1}^T A p_{k-1})"),
            MathTex(r"x_k = x_{k-1} + \alpha_k p_{k-1}"),
            MathTex(r"r_k = r_{k-1} - \alpha_k A p_{k-1}"),
            MathTex(r"\beta_k = (r_k^T r_k) / (r_{k-1}^T r_{k-1})"),
            MathTex(r"p_k = r_k + \beta_k p_{k-1}"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).scale(.6).to_corner(DL, buff=1).shift(DOWN * .2)
        self.algo[2:].shift(RIGHT * .6)
        self.rect = SurroundingRectangle(self.algo)
        self.desc = Tex(r"{\kaishu 在互相$A$-conjugate的方向中，\\每步干掉$e_k=x^*-x_k$的一个分量.}", color=YELLOW, font_size=30).next_to(self.rect, UP)
        
        # 右下角的评论
        comment_font_size = 35
        self.comments = VGroup(
            Tex(r"1. $p_0=b-Ax_0=-\nabla f(x_0)$: ``共轭梯度''", font_size=comment_font_size),
            Tex(r"2. 第$k$步在$\mathcal{K}_k$中{{minimize $\lVert e^k \rVert_A$}}", font_size=comment_font_size).set_color_by_tex("min", YELLOW),
            Tex(r"3. 每步一次$Ap_{k-1}$: 稠密$2n^2$，稀疏$O(n)$", font_size=comment_font_size),
            Tex(r"4. 只需上一步信息，空间占用小", font_size=comment_font_size),
            Tex(r"5. 收敛速度与特征值分布有关:", font_size=comment_font_size),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_corner(DR, buff=1).shift(UP)

        self.rel_err = MathTex(r"{\lVert e_k \rVert_A\over\lVert e_0\rVert_A}\leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k,~\kappa=\frac{\lambda_{\max}}{\lambda_{\min}}", color=BLUE, font_size=30).next_to(self.comments, DOWN)

        self.bg = RoundedRectangle(
            corner_radius=0.2,
            fill_color=GOLD,
            fill_opacity=0.2,
            stroke_width=0
        ).surround(VGroup(self.comments, self.rel_err), buff=.5).stretch(1.2, dim=1)

    def construct(self):
        self.play(Write(self.i_conj))
        self.wait()
        self.play(Write(self.a_conj))
        self.play(Write(self.aside))
        self.wait()
        self.play(Write(self.a_norm))
        self.play(Write(self.aside2))
        self.wait()
        self.play(Write(self.desc))
        self.wait()
        
        # 显示算法伪代码
        self.play(FadeIn(self.algo), Create(self.rect))
        self.wait()
        
        # 显示右下角的评论
        self.play(FadeIn(self.bg))
        self.bring_to_back(self.bg)
        for comment in self.comments:
            self.play(Write(comment))
            self.wait()
        self.play(Write(self.rel_err))
        self.wait()


class Preconditioning(ThreeDScene):
    def __init__(self):
        super().__init__()
        # 标题
        self.title = Title("Preconditioning")
        
        # 问题描述
        self.problem = MathTex(r"Ax = b", r"~\Longrightarrow~ ", r"M^{-1}", r"Ax=", r"M^{-1}", "b").next_to(self.title, DOWN, buff=0.25).set_color_by_tex_to_color_map({'M': YELLOW})
        
        
    def construct(self):
        self.add_fixed_in_frame_mobjects(self.title)
        self.wait()
        self.play(Write(self.problem))
        self.wait()
        self.camera.add_fixed_in_frame_mobjects(self.problem)
        
        # 2. 可视化地形对比
        self.visualize_terrain()
        
        # 3. 迭代速度对比
        self.compare_convergence()
        
        # 4. 常见预条件器
        self.show_preconditioners()
        
    def visualize_terrain(self):
        """可视化原始A和预条件后的地形"""
        # self.play(self.title.animate.scale(0.7).to_corner(UL))
        
        # 创建两个3D坐标系
        axes = ThreeDAxes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            z_range=[0, 10, 2],
            x_length=4,
            y_length=4,
            z_length=3,
        ).shift(IN * 2)
        
        # 创建地形曲面 - 病态矩阵 (高条件数)
        def bad_surface(u, v):
            return axes.c2p(u, v, 5 * u**2 + 0.1 * v**2)
        
        surface_bad = Surface(
            bad_surface,
            u_range=[-1.5, 1.5],
            v_range=[-1.5, 1.5],
            fill_opacity=0.7,
            checkerboard_colors=[RED_D, RED_E]
        )
        
        # 创建地形曲面 - 良态矩阵 (低条件数)
        def good_surface(u, v):
            return axes.c2p(u, v, u**2 + v**2)
        
        surface_good = Surface(
            good_surface,
            u_range=[-1.5, 1.5],
            v_range=[-1.5, 1.5],
            fill_opacity=0.7,
            checkerboard_colors=[GREEN_D, GREEN_E]
        )
        
        # 设置相机角度
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        
        self.play(FadeIn(axes), Create(surface_bad),)
        self.wait()
        self.play(ReplacementTransform(surface_bad, surface_good))
        self.wait()
        
        # 清屏
        self.play(FadeOut(VGroup(axes, surface_good)))        
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)

    def compare_convergence(self):
        """对比迭代速度"""
        # 创建坐标轴
        axes = Axes(
            x_range=[0, 52, 10],
            y_range=[0, 1.1, 0.2],
            x_length=10,
            y_length=5,
            axis_config={"include_tip": True},
            tips=True
        ).shift(DOWN * .5)
        
        x_label = axes.get_x_axis_label(MathTex("k", font_size=40))
        y_label = axes.get_y_axis_label(MathTex(r"\lVert r_k\rVert", font_size=40), edge=LEFT, direction=LEFT)
        
        self.play(FadeIn(axes), FadeIn(x_label), FadeIn(y_label))
        
        # 原始迭代法 - 慢收敛
        original_points = [
            axes.c2p(k, 0.95 ** k) for k in range(51)
        ]
        original_line = VMobject(color=RED, stroke_width=6)
        original_line.set_points_as_corners(original_points)
        original_dots = VGroup(*[Dot(p, color=RED, radius=0.1) for p in original_points[::3]])
        
        # 预条件迭代法 - 快收敛
        precond_points = [
            axes.c2p(k, 0.7 ** k) for k in range(51)
        ]
        precond_line = VMobject(color=GREEN, stroke_width=6)
        precond_line.set_points_as_corners(precond_points)
        precond_dots = VGroup(*[Dot(p, color=GREEN, radius=0.1) for p in precond_points[::3]])
        
        # 图例
        legend_original = Tex("not preconditioned", color=original_line.get_color(), font_size=30).move_to(axes.c2p(30, 0.4))
        self.play(
            Create(original_line),
            Create(original_dots),
            Write(legend_original),
            run_time=1
        )
        self.wait()
        
        legend_precond = Tex("preconditioned", color=precond_line.get_color(), font_size=30).move_to(axes.c2p(12, 0.2))
        self.play(
            Create(precond_line),
            Create(precond_dots),
            Write(legend_precond),
            run_time=1
        )
        self.wait()
        
        # 清屏
        self.play(
            FadeOut(VGroup(axes, x_label, y_label, original_line, original_dots, 
                          precond_line, precond_dots, legend_original, legend_precond))
        )
        
    def show_preconditioners(self):
        """展示几种常见的预条件器"""
        
        # 预条件器列表
        preconditioners = VGroup(
            VGroup(
                Tex("1. Jacobi / Gauss-Seidel / SOR / ...", font_size=32, color=BLUE),
                MathTex(r"M = D,~M = D - L,~\ldots", font_size=28, color=YELLOW)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.2),
            
            VGroup(
                Tex("2. Incomplete LU / Cholesky", font_size=32, color=BLUE),
                MathTex(r"M \approx LU,~M \approx LL^T", font_size=28, color=YELLOW)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.2),
            
            VGroup(
                Tex("3. Multigrid", font_size=32, color=BLUE),
                Tex(r"多层网格粗化", font_size=28, color=YELLOW)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.2),
            
            VGroup(
                Tex("4. Machine Learning", font_size=32, color=BLUE),
                MathTex(r"M = \text{neural network}(A)", font_size=28, color=YELLOW)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.2),

            Tex("...", color=BLUE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(self.problem, DOWN, buff=.5)
        
        # 逐个显示
        for precond in preconditioners:
            self.play(FadeIn(precond, shift=RIGHT * 0.5))
            self.wait(0.8)
        
        self.wait()
        

class DontInvert(Scene):
    def construct(self):
        wrong = Tex(r"``只要''求出 $A^{-1}$ ``就''能解方程组").to_edge(UP, buff=.5)
        cross = Cross(wrong)

        font_size = 40
        drawbacks = VGroup(
            Tex("1. 计算量大", font_size=font_size),
            Tex("2. 数值稳定性差", font_size=font_size),
            Tex("3. 破坏稀疏性", font_size=font_size)
        ).set_color(YELLOW).arrange(DOWN, aligned_edge=LEFT, buff=.4).next_to(wrong, DOWN, buff=.5)

        self.play(Write(wrong))
        self.play(Create(cross))
        self.wait()

        for d in drawbacks:
            self.play(Write(d))
            self.wait()

        # self.play(FadeOut(drawbacks))
        # self.wait()

        # MATLAB 代码示例
        matlab_label = Tex("MATLAB", font_size=45, color=BLUE).shift(LEFT * 3 + DOWN * .25)
        matlab_correct = Code(
            code_string=r"x = A \ b;",
            language="matlab",
            background="window",
        ).scale(0.8).next_to(matlab_label, DOWN, buff=0.5)
        
        matlab_wrong = Code(
            code_string="x = inv(A) * b;",
            language="matlab",
            background="window",
        ).scale(0.8).next_to(matlab_correct, DOWN, buff=0.3)
        
        matlab_cross = Cross(matlab_wrong, stroke_color=RED, stroke_width=6)

        # NumPy 代码示例
        numpy_label = Tex("NumPy", font_size=45, color=ORANGE).move_to(matlab_label).shift(RIGHT * 6)
        numpy_correct = Code(
            code_string="x = np.linalg.solve(A, b)",
            language="python",
            background="window",
        ).scale(0.8).next_to(numpy_label, DOWN, buff=0.5)
        
        numpy_wrong = Code(
            code_string="x = np.linalg.inv(A) @ b",
            language="python",
            background="window",
        ).scale(0.8).next_to(numpy_correct, DOWN, buff=0.3)
        
        numpy_cross = Cross(numpy_wrong, stroke_color=RED, stroke_width=6)

        self.play(FadeIn(matlab_label), FadeIn(matlab_correct))
        self.wait()
        self.play(FadeIn(numpy_label), FadeIn(numpy_correct))
        self.wait()
        self.play(
            FadeIn(matlab_wrong), Create(matlab_cross),
            FadeIn(numpy_wrong), Create(numpy_cross)
            )
        self.wait()


class Generalization(Scene):
    def construct(self):
        font_size = 80
        v = VGroup(
            MathTex(r"A^{-1}", 'b', font_size=font_size),
            MathTex(r"A_1A_2A_3", 'b', font_size=font_size),
            MathTex(r"(A^3+3A^2+A)", 'b', font_size=font_size),
            MathTex(r"\mathrm{e}^A", 'b', font_size=font_size),
        ).arrange(DOWN, buff=.75)
        for t in v:
            t.set_color_by_tex_to_color_map({'A': RED, 'b': YELLOW})
        self.add(*[t[0] for t in v])
        self.wait()
        self.play(
            AnimationGroup(
                Write(t[1]) for t in v
            )
        )
        self.wait()


class Thumbnail(Scene):
    def construct(self):
        font_size = 160
        stroke_width = 9
        v = VGroup(
            MathTex(r"\text{求}~ {{A^{-1}}} {{b}}", font_size=font_size, stroke_width=stroke_width),
            MathTex(r"\text{而非}~ {{A^{-1}}}{{\times}} {{b}}", font_size=font_size, stroke_width=stroke_width)
        ).arrange(DOWN, buff=.8)
        for t in v:
            t.set_color_by_tex_to_color_map({'A': YELLOW, 'b': BLUE, 'times': RED})
        # v[1][0][:2].set_color(RED)
        self.add(v)