import numpy as np
from numpy import linalg
from math import sqrt

ROUND = True
if ROUND:    
    RN = 3 # ndigits to round to

class RedundantArguments(SyntaxError):
    def __init__(self):
        pass


class Point:
    def __init__(self, x: float, y: float, z: float, name=''):
        if type(x) == np.ndarray:
            [x, y, z] = list(map(lambda x: x.item(), [x, y, z]))
        self.x = x
        self.y = y
        self.z = z
        self.name = name

    def __repr__(self):
        a = [self.x, self.y, self.z] if not ROUND else [round(self.x, RN), round(self.y, RN), round(self.z, RN)]
        return f'{self.name} ({a[0]}, {a[1]}, {a[2]})'

    __str__ = __repr__

    def __eq__(self, other):
        if other is None:
            return self is None
        return [self.x, self.y, self.z] == [other.x, other.y, other.z]

    def __mul__(self, other):
        return middle_point(self, other)


class Vector:
    def __init__(self, pt1: Point = None, pt2: Point = None,
                 x: float = None, y: float = None, z: float = None, name: str = ''):

        k = False
        for i in [pt1, pt2, x, y, z]:
            if i:
                k = True
                break

        if not k:
            if [x, y, z] == [0, 0, 0]:
                print('warn: null vector created(' + name + ')')
            else:
                raise SyntaxError
        if pt1 and pt2:
            if x or y or z:
                raise RedundantArguments
            if not name:
                self.name = str(pt1.name) + str(pt2.name)
            else:
                self.name = name
            self.x = pt2.x - pt1.x
            self.y = pt2.y - pt1.y
            self.z = pt2.z - pt1.z
            self.mat = np.array([[self.x], [self.y], [self.z]])
            self.norm = linalg.norm(self.mat)
            return
        if [x, y, z] != [None, None, None]:
            if pt1 or pt2:
                raise RedundantArguments

            self.name = name
            self.x = x
            self.y = y
            self.z = z
            self.mat = np.array([[self.x], [self.y], [self.z]])
            self.norm = linalg.norm(self.mat)
        else:
            raise SyntaxError

    def __repr__(self):
        i = '\t' * (len(self.name) // 4 + 1)
        a = [self.x, self.y, self.z] if not ROUND else [round(self.x, RN), round(self.y, RN), round(self.z, RN)]
        return f'{self.name}\t{a[0]}\n{i}{a[1]}\n{i}{a[2]}'

    def __eq__(self, other):
        return self.norm == other.norm and vectors_collinear(self, other)

    def __add__(self, other):
        return Vector(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def __sub__(self, other):
        return Vector(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)

    def __mul__(self, other):   # Returns the dot product
        return scalar(self, other)
    __rmul__ = __mul__


class Plane:
    def __init__(self, name: str = 'P', pt1: Point = None, pt2: Point = None, pt3: Point = None,
                 vec1: Vector = None, vec2: Vector = None,
                 a: float = None, b: float = None, c: float = None, d: float = None,
                 normal_vector: Vector = None):

        self.a = None
        self.b = None
        self.c = None
        self.d = None
        k = False
        for i in [pt1, pt2, pt3, vec1, vec2, a, b, c, d]:
            if i is not None:
                k = True
                break

        if not k:
            raise SyntaxError

        self.name = name

        if normal_vector and pt1:
            if (pt2 or pt3) or (a or b or c or d) or (vec2 or vec1):
                raise RedundantArguments

            self.a = normal_vector.x
            self.b = normal_vector.y
            self.c = normal_vector.z
            self.normal_vector = normal_vector
            self.d = -normal_vector.x * pt1.x - normal_vector.y * pt1.y - normal_vector.z * pt1.z
            return

        tr = True

        if [pt2, pt3] != [None, None]:
            if pt1 is None:
                raise SyntaxError
            vec1 = Vector(pt1, pt2)
            vec2 = Vector(pt1, pt3)
            tr = False

        if [vec1, vec2, pt1] != [None, None, None]:
            if ((pt2 or pt3) and tr) or (a or b or c or d):
                raise RedundantArguments

            self.vec1 = vec1
            self.vec2 = vec2

            if vectors_collinear(vec1, vec2):
                print('Vectors collinear, cannot define plane')
                return

            self.a = vec1.y * vec2.z - vec1.z * vec2.y
            self.b = vec1.z * vec2.x - vec1.x * vec2.z
            self.c = vec1.x * vec2.y - vec1.y * vec2.x
            self.d = pt1.x * (vec1.z * vec2.y - vec1.y * vec2.z) + pt1.y * (vec1.x * vec2.z - vec1.z * vec2.x) + \
                     pt1.z * (vec1.y * vec2.x - vec1.x * vec2.y)

            if self.d == 0:
                if self.a == 0:
                    if self.b == 0:
                        self.c = 1

                    elif self.c == 0:
                        self.b = 1

                elif [self.c, self.b] == [0, 0]:
                    self.a = 1

            self.normal_vector = Vector(name=f'N_{self.name}', x=self.a, y=self.b, z=self.c)
            return
        elif [a, b, c, d] != [None, None, None, None]:
            if ((pt1 or pt2 or pt3) and tr) or (vec1 or vec2):
                raise RedundantArguments

            if [a, b, c, d] == [0, 0, 0, 0]:
                self.a = None
                self.b = None
                self.c = None
                self.d = None
                print('Impossible plane equation')
                return

            self.vec1 = None
            self.vec2 = None

            self.a = a
            self.b = b
            self.c = c
            self.d = d
            self.normal_vector = Vector(name=f'N_{self.name}', x=self.a, y=self.b, z=self.c)
        else:
            raise SyntaxError

    def __repr__(self):
        if [self.a, self.b, self.c, self.d] == [None, None, None, None]:
            print('Plane undefined')
            return
        if self.d == 0:
            sv = ''
        else:
            sv = ' ' + dress_num(self.d)
        return f'{self.name}: {eq_mk("x", self.a) + eq_mk("y", self.b) + eq_mk("z", self.c) + sv} = 0'

    def __eq__(self, other):
        return relation_two_planes(self, other) == 'confendus'


class Line:
    def __init__(self, parameter: str = 'al', pt: Point = None, vec: Vector = None, pt2: Point = None, name: str = '',
                 p1: Plane = None, p2: Plane = None):
        tr = True
        self.name = name
        if pt and pt2 and tr:
            vec = Vector(pt, pt2)
            tr = False
            if not name:
                self.name = f'({pt.name}{pt2.name})'

        if p1 and p2:
            k = p2.c - (p1.c * p2.b) / p1.b
            x = 0
            z = (((p1.d * p2.b) / p1.d) - p2.d) / k
            y = - p1.d - (p1.d * p2.b * p1.c - p2.d) / (p1.c * k)

            a = 1
            b = ((p2.b * p1.a * p1.c) / p1.b - p1.c * p2.a) / k
            c = (((p2.b * p1.a) / p1.b) - p2.a) / k

            pt = Point(name='PT{', x=x, y=y, z=z)
            vec = Vector(name=f'vec_{self.name}', x=a, y=b, z=c)

        if not np.any(vec.mat):
            raise Exception('Cannot make line from null vector')

        if pt and vec:
            if pt2 and tr:
                raise RedundantArguments
            self.parameter = parameter
            self.pt = pt
            self.vec = vec
            self.a = np.array([[pt.x], [pt.y], [pt.z]])
            self.b = vec.mat

    def point_from_line(self, parameter: float) -> Point: 
        c = self.a + parameter * self.b
        pt = Point(name='PT{' + f'{self.parameter} = {parameter}' + '}', x=c[0], y=c[1], z=c[2])
        return pt

    def __repr__(self):
        i = '\t' * (len(self.name) // 4 + 1)
        return f'{self.name}\tx = {eq_mk_line(self.a[0], self.b[0], self.parameter)}\n{i}' \
               f'y = {eq_mk_line(self.a[1], self.b[1], self.parameter)}\n{i}' \
               f'z = {eq_mk_line(self.a[2], self.b[2], self.parameter)}'

    def __eq__(self, other):
        return vectors_collinear(self.vec, other.vec) and point_in_line(self.pt, other)


class Diameter:
    def __init__(self, pt1: Point, pt2: Point):
        self.pt1 = pt1
        self.pt2 = pt2
        self.center = middle_point(self.pt1, self.pt2)
        self.length = distance_two_points(self.pt1, self.pt2)

    def __repr__(self):
        return f'[{self.pt1.name}{self.pt2.name}]'


class Circle:
    def __init__(self, center: Point, r: float, name: str = 'C'):
        self.center = center
        self.r = r
        self.name = name

    def __repr__(self):
        r = self.r if not ROUND else round(self.r, RN)
        return f'{self.name}({self.center.name}, {r})'

    __str__ = __repr__
        

class Sphere:
    def __init__(self, name: str = 'S', center: Point = None, r: float = None, d: Diameter = None):
        if [center, r] != [None, None] and d is not None:
            raise RedundantArguments

        if [center, r, d] == [None, None, None]:
            raise SyntaxError

        self.name = name

        if [center, r] != [None, None]:
            self.center = center
            self.r = r

        elif d is not None:
            self.center = d.center
            self.r = d.length / 2

        if self.r < 0:
            raise AttributeError

        self.equation = eq_mk_sphere(-self.center.x, 'x') + eq_mk_sphere(-self.center.y, 'y') + eq_mk_sphere(-self.center.z, 'z') \
            + f' = {self.r ** 2}'

    def __repr__(self):
        return f'{self.name}({self.center}, {self.r})'


def scalar(vec1: Vector, vec2: Vector):
    return np.tensordot(vec1.mat, vec2.mat)


def vectors_collinear(vec1: Vector, vec2: Vector):
    a = np.array([[vec1.x, vec2.x], [vec1.y, vec2.y]])
    b = np.array([[vec1.x, vec2.x], [vec1.z, vec2.z]])
    c = np.array([[vec1.y, vec2.y], [vec1.z, vec2.z]])
    det_a = linalg.det(a)
    det_b = linalg.det(b)
    det_c = linalg.det(c)
    #    if list(map(lambda x: round(x), [det_a, det_b, det_c]))
    if [det_a, det_b, det_c] == [0, 0, 0]:
        return True

    return False


def vectors_orthogonal(vec1: Vector, vec2: Vector):
    if scalar(vec1, vec2) == 0:
        return True
    return False


def dress_num(n):
    if n == 1:
        return '+ '

    if n == -1:
        return '- '
    if n < 0:
        return '- ' + str(abs(n))

    elif n != 0:
        return '+ ' + str(n)

    return n


def eq_mk(var: str = '', n=0.):
    if n == 0:
        return ''

    if ROUND:
        n = round(n, RN)

    if var == 'x' and n > 0:
        if n == 1:
            return 'x'
        return str(n) + 'x'

    return ' ' + dress_num(n) + var


def eq_mk_line(a, b, par: str):
    if [type(a), type(b)] == [np.ndarray, np.ndarray]:
        a = a.item()
        b = b.item()

    if [a, b] == [0, 0]:
        return '0'

    if ROUND:
        a, b = round(a, RN), round(b, RN)

    if a == 0:
        if b == 1:
            return par
        if b == -1:
            return '-' + par

        return str(b) + par

    if b == 0:
        return str(a)

    if b == 1:
        return str(a) + ' + ' + par

    if b == -1:
        return str(a) + ' - ' + par

    if b > 0:
        return str(a) + ' + ' + str(b) + par

    if b < 0:
        return str(a) + ' - ' + str(abs(b)) + par


def eq_mk_sphere(n: float, var: str):
    if ROUND:
        n = round(n, RN)
    if var == 'x':
        if n == 0:
            return 'x²'

        if n < 0:
            return f'(x - {abs(n)})²'

        if n > 0:
            return '(x + ' + str(n) + ')²'

    if n == 0:
        return ' + ' + var + '²'

    if n < 0:
        return f' + ({var} - {abs(n)})²'

    if n > 0:
        return f' + ({var} + {n})²'


def planes_parallel(p1: Plane, p2: Plane):
    if vectors_collinear(p1.normal_vector, p2.normal_vector):
        return True
    return False


def middle_point(pt1: Point, pt2: Point):
    return Point(x=(pt1.x + pt2.x) / 2, y=(pt1.y + pt2.y) / 2, z=(pt1.z + pt2.z) / 2, name=f'{pt1.name}*{pt2.name}')


def point_in_line(pt: Point, line: Line) -> bool:
    i = None

    for v, a, p in zip([line.vec.x, line.vec.y, line.vec.z], [line.pt.x, line.pt.y, line.pt.z], [pt.x, pt.y, pt.z]):

        if v == 0:
            if p != a:
                return False

            continue

        if i is None:
            i = (p - a) / v

        else:
            if i != (p - a) / v:
                return False

    return True


def point_in_plane(pt: Point, p: Plane):
    if pt.x * p.a + pt.y * p.b + pt.z * p.c + p.d == 0:
        return True
    return False


def relation_two_lines(l1: Line, l2: Line):
    if vectors_collinear(l1.vec, l2.vec):
        if point_in_line(l1.pt, l2):
            return 'merged'
        return 'parallel'

    a = np.array([[l1.vec.x, -l2.vec.x], [l1.vec.y, -l2.vec.y]])
    b = np.array([[l2.pt.x - l1.pt.x], [l2.pt.y - l1.pt.y]])
    s = linalg.solve(a, b)

    if l1.vec.z * s[0] - l2.vec.z * s[1] == l2.pt.z - l1.pt.z:
        k = l1.point_from_line(s.item(0))

        if vectors_orthogonal(l1.vec, l2.vec):
            return 'perpendicular', k

        return 'secant', k

    return 'non secant'


def relation_two_planes(p1: Plane, p2: Plane):
    if [p1.a, p1.b, p1.c, p1.d] == [p2.a, p2.b, p2.c, p2.d]:
        return 'confendus'

    if planes_parallel(p1, p2):
        return 'parallel'

    if vectors_orthogonal(p1.normal_vector, p2.normal_vector):
        return 'Perpendicular'

    return 'secants'


def relation_plane_line(p: Plane, l: Line, proj=False):
    if vectors_orthogonal(p.normal_vector, l.vec):
        if point_in_plane(l.pt, p):
            return 'confendus'
        return 'parallel'

    x, y, z = l.pt.x, l.pt.y, l.pt.z
    e, f, g = l.vec.x, l.vec.y, l.vec.z
    a, b, c, d = p.a, p.b, p.c, p.d

    alpha = (-a * x - b * y - c * z - d) / (a * e + b * f + c * g)

    k = l.point_from_line(alpha)

    if proj:
        k.name = 'H'

    if vectors_collinear(l.vec, p.normal_vector):
        return 'perpendicular', k

    return 'secant', k


def relation_plane_sphere(p: Plane, s: Sphere):
    d = distance_point_plane(s.center, p)
    r = s.r
    if d > r:
        return 'disjoints'

    if d == r:
        return 'tangents', orthogonal_projection_point_plane(s.center, p)

    if point_in_plane(s.center, p):
        return 'mediator', Circle(s.center, r)

    return 'secants', Circle(orthogonal_projection_point_plane(s.center, p), sqrt(r**2 - d**2))


def mediator_plane(pt1: Point, pt2: Point):
    i = middle_point(pt1, pt2)
    nv = Vector(pt1, pt2)
    return Plane(name=f'Pmed[{pt1.name}{pt2.name}]', pt1=i, normal_vector=nv)


def distance_two_points(pt1: Point, pt2: Point):
    vec = Vector(pt1, pt2)
    return vec.norm


def distance_point_plane(pt: Point, p: Plane):
    if point_in_plane(pt, p):
        return 0

    a, b, c, d = p.a, p.b, p.c, p.d
    x, y, z = pt.x, pt.y, pt.z
    return abs(a * x + b * y + c * z + d) / p.normal_vector.norm


def distance_point_line(pt: Point, l: Line):
    if point_in_line(pt, l):
        return 0

    h = orthogonal_projection_point_line(pt, l)
    v = Vector(h, pt)

    return v.norm


def orthogonal_projection_point_plane(pt: Point, p: Plane):
    if point_in_plane(pt, p):
        return pt

    li = Line(pt=pt, vec=p.normal_vector)

    return relation_plane_line(p, li, True)[1]


def orthogonal_projection_point_line(pt: Point, l: Line):
    if point_in_line(pt, l):
        return pt

    p = Plane(normal_vector=l.vec, pt1=pt)

    return relation_plane_line(p, l, True)[1]


if __name__ == '__main__':
    B = Point(0,0,2)
    C = Point(1,2,0)
    A = Point(0,2,0)
    O = Point(0,0,0)
    I = O*A
    BC = Line(pt=B,pt2=C, name='(BC)')
    H = orthogonal_projection_point_line(I, BC)
    P = mediator_plane(H, I)
    S = Sphere(d=Diameter(O, A))
    print(S)
    C = relation_plane_sphere(P, S)[1]
    print(C.r)
    print(H)
