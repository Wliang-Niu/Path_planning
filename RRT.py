import math
import random

import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera

class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        创建节点
        """

        def __init__(self, x, y):
            self.x = x  # 节点坐标
            self.y = y
            self.path_x = []  # 路径，作为画图的数据
            self.path_y = []
            self.parent = None  # 父节点

    class AreaBounds:
        """区域大小
        """

        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=3.0,
                 goal_sample_rate=5,
                 max_iter=1500,
                 play_area=None,
                 robot_radius=0.0,
                 ellipse_inflate=1.8,
                 ):
        """
        Setting Parameter

        start:起点 [x,y]
        goal:目标点 [x,y]
        obstacleList:障碍物位置列表 [[x,y,size],...]
        rand_area: 采样区域 x,y ∈ [min,max]
        play_area: 约束随机树的范围 [xmin,xmax,ymin,ymax]
        robot_radius: 机器人半径
        expand_dis: 扩展的步长
        goal_sample_rate: 采样目标点的概率，百分制.default: 5，即表示5%的概率直接采样目标点
        ellipse_inflate: 椭球聚焦采样的长轴膨胀系数，None 表示退回均匀采样

        """
        self.start = self.Node(start[0], start[1])  # 根节点
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.robot_radius = robot_radius
        self.ellipse_inflate = ellipse_inflate
        # 预计算起点-目标为焦点的椭球几何，避免每次采样重复计算
        self.ellipse = self._build_ellipse() if ellipse_inflate is not None else None

    def planning(self, animation=True, camera=None):
        """
        rrt path planning

        animation: flag for animation on or off

        camera: 是否保存动图
        """

        # 将起点作为根节点，加入到随机树的节点集合中。
        self.node_list = [self.start]
        for i in range(self.max_iter):
            # 从可行区域内随机选取一个节点
            rnd_node = self.sample_free()

            # 已生成的树中利用欧氏距离判断距离最近的点。
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            # 从x_{near}与x_{rand}的连线方向上扩展固定步长u，得到新节点 x_{new}
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # 如果在可行区域内，且x_{near}与x_{new}之间无障碍物
            if self.is_inside_play_area(new_node, self.play_area) and \
                    self.obstacle_free(new_node, self.obstacle_list, self.robot_radius):
                self.node_list.append(new_node)

            # 如果此时得到的节点x_new到目标点的距离小于扩展步长，则直接将目标点作为x_rand。
            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.obstacle_free(final_node, self.obstacle_list, self.robot_radius):
                    # 返回最终路径
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node, camera)

        return None  # cannot find path

    def plan_with_retry(self, retries=5, animation=True, camera=None):
        """planning 返回 None 时自动换新随机过程重试，提高有限迭代下的成功率"""
        for k in range(retries):
            path = self.planning(animation=animation, camera=camera)
            if path is not None:
                print(f"第 {k + 1} 次尝试成功")
                return path
        return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """连线方向扩展固定步长查找x_new

        Args:
            from_node (_type_): x_near
            to_node (_type_): x_rand
            extend_length (_type_, optional): 扩展步长u. Defaults to float("inf").

        Returns:
            _type_: _description_
        """
        # 利用反正切计算角度, 然后利用角度和步长计算新坐标
        d, theta = self.calc_distance_and_angle(from_node, to_node)

        # 如果x_{near}与x_{rand}间的距离小于步长，则直接将x_{rand}作为新节点x_{new}
        if extend_length >= d:
            new_x = to_node.x
            new_y = to_node.y
        else:
            new_x = from_node.x + math.cos(theta) * extend_length
            new_y = from_node.y + math.sin(theta) * extend_length
        new_node = self.Node(new_x, new_y)
        new_node.path_x = [from_node.x]
        new_node.path_y = [from_node.y]
        new_node.path_x.append(new_x)
        new_node.path_y.append(new_y)

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        """生成路径
        Args:
            goal_ind (_type_): 目标点索引

        Returns:
            _type_: _description_
        """
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        """计算(x,y)离目标点的距离
        """
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def _build_ellipse(self):
        """以起点、目标为焦点构建聚焦采样椭球，返回 (a, b, cx, cy, ang)"""
        sx, sy = self.start.x, self.start.y
        gx, gy = self.end.x, self.end.y
        d = math.hypot(gx - sx, gy - sy)
        c = d / 2.0
        a = c * self.ellipse_inflate
        b = math.sqrt(max(a * a - c * c, 1e-9))
        cx = (sx + gx) / 2.0
        cy = (sy + gy) / 2.0
        ang = math.atan2(gy - sy, gx - sx)
        return a, b, cx, cy, ang

    def sample_free(self):
        # 以 goal_sample_rate% 的概率直接采样目标点
        if random.randint(0, 100) <= self.goal_sample_rate:
            return self.Node(self.end.x, self.end.y)
        if self.ellipse is None:
            # 未启用椭球：退回活动区域/采样区域内均匀采样
            if self.play_area is not None:
                x_lo, x_hi = self.play_area.xmin, self.play_area.xmax
                y_lo, y_hi = self.play_area.ymin, self.play_area.ymax
            else:
                x_lo, x_hi = self.min_rand, self.max_rand
                y_lo, y_hi = self.min_rand, self.max_rand
            return self.Node(random.uniform(x_lo, x_hi),
                             random.uniform(y_lo, y_hi))
        # 椭球内拒绝采样，并保证落在活动区域内
        a, b, cx, cy, ang = self.ellipse
        while True:
            ux = random.uniform(-a, a)
            uy = random.uniform(-b, b)
            if (ux / a) ** 2 + (uy / b) ** 2 > 1.0:
                continue
            rnd_x = cx + ux * math.cos(ang) - uy * math.sin(ang)
            rnd_y = cy + ux * math.sin(ang) + uy * math.cos(ang)
            if self.is_inside_play_area(self.Node(rnd_x, rnd_y), self.play_area):
                return self.Node(rnd_x, rnd_y)

    def draw_graph(self, rnd=None, camera=None):
        if camera == None:
            plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        # 画随机点
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
            if self.robot_radius > 0.0:
                self.plot_circle(rnd.x, rnd.y, self.robot_radius, '-r')
        # 画已生成的树
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        # 画障碍物
        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        # 如果约定了可行区域，则画出可行区域
        if self.play_area is not None:
            plt.plot([self.play_area.xmin, self.play_area.xmax,
                      self.play_area.xmax, self.play_area.xmin,
                      self.play_area.xmin],
                     [self.play_area.ymin, self.play_area.ymin,
                      self.play_area.ymax, self.play_area.ymax,
                      self.play_area.ymin],
                     "-k")

        # 画出起点和目标点
        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)
        if camera != None:
            camera.snap()

    # 静态方法无需实例化，也可以实例化后调用，静态方法内部不能调用self.的变量
    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def is_inside_play_area(node, play_area):

        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin or node.x > play_area.xmax or \
                node.y < play_area.ymin or node.y > play_area.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok

    @staticmethod
    def obstacle_free(node, obstacleList, robot_radius):
        if node is None:
            return False
        # 遍历这条边包含的每一段（当前 path_x/path_y 有两个点，即一条线段）
        for i in range(len(node.path_x) - 1):
            x1, y1 = node.path_x[i], node.path_y[i]
            x2, y2 = node.path_x[i + 1], node.path_y[i + 1]
            for (ox, oy, size) in obstacleList:
                d_sq = RRT.point_to_segment_dist_sq(
                    ox, oy, x1, y1, x2, y2)
                if d_sq <= (size + robot_radius) ** 2:
                    return False  # 碰撞
        return True  # 安全

    @staticmethod
    def point_to_segment_dist_sq(px, py, x1, y1, x2, y2):
        """点(px,py) 到线段 (x1,y1)-(x2,y2) 的最短距离平方"""
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy
        # 线段退化为一个点时，直接算点到点距离
        if length_sq < 1e-12:
            tdx, tdy = px - x1, py - y1
            return tdx * tdx + tdy * tdy
        # 投影参数
        t = ((px - x1) * dx + (py - y1) * dy) / length_sq
        if t < 0.0:
            cx, cy = x1, y1
        elif t > 1.0:
            cx, cy = x2, y2
        else:
            cx = x1 + t * dx
            cy = y1 + t * dy
        tdx, tdy = px - cx, py - cy
        return tdx * tdx + tdy * tdy

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        """计算两个节点间的距离和方位角

        Args:
            from_node (_type_): _description_
            to_node (_type_): _description_

        Returns:
            _type_: _description_
        """
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

def main(gx=6.0, gy=10.0):
    print("start " + __file__)
    fig = plt.figure(1)

    camera = Camera(fig)  # 保存动图时使用
    # camera = None  # 不保存动图时，camara为None
    show_animation = True
    # ====Search Path with RRT====
    obstacleList = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2),
        (8, 10, 1),
        (6, 12, 1),
    ]  # [x,y,size(radius)]
    # Set Initial parameters
    rrt = RRT(
        start=[0, 0],
        goal=[gx, gy],
        rand_area=[-2, 15],
        obstacle_list=obstacleList,
        play_area=[-2, 12, 0, 14],
        robot_radius=0.8
    )
    path = rrt.plan_with_retry(animation=show_animation, camera=camera)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            rrt.draw_graph(camera=camera)
            plt.grid(True)
            plt.pause(0.01)
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            if camera != None:
                camera.snap()
                animation = camera.animate()
                animation.save('trajectory.gif')
            plt.figure(2)
            plt.axis("equal")
            plt.axis([-2, 15, -2, 15])
            plt.grid(True)
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.show()


if __name__ == '__main__':
    main()