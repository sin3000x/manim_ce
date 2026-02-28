from manim import *
from manim.scene.vector_space_scene import X_COLOR, Y_COLOR

# 让本文件内所有 Tex/MathTex 默认使用 ctex，无需逐个传 tex_template
config.tex_template = TexTemplateLibrary.ctex

# 设置全局默认 stroke_width
Tex.set_default(stroke_width=2)
MathTex.set_default(stroke_width=2)

# 设置全局默认 Matrix h_buff
Matrix.set_default(h_buff=0.8)


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
