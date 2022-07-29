import bpy
import json
import bmesh
import numpy as np
import mathutils
import math

 
def xyz(val=(0, 0, 0), z_in=0):
    '''Entran coordenadas en 2D o en 3D y salen en 3D'''
    try:
        z = float(val[2])
    except Exception as inst:
        z = z_in  # val[1]
    return (float(val[0]), float(val[1]), z)

# El siguiente diccionario tiene algunos valores por defecto
defecto = {
    'llamad': 'alguna_cosa',
    'de': 'white',
    'en': (0, 0, 0),
    'escalado': 0.3,
    'alpha': 0.25,
    'brillo': 2,
    'ancho_linea': 0.02,
    'espacio_entre_palabras': 0.4,
    'espacio_entre_renglones': 1,
    'algunos_colores': ('cyan', 'magenta', 'yellow', 'black', 'white', 'blue', 'green', 'red'),
    'ejes_0': (-8, -8, -8),
    'ejes_1': (8, 8, 8),
    'radio_punto': 0.1,
}

objs={}

def args(kwargs, arg, default=None):
    ''''Si en `kwargs` no está la llave la busca en `defecto`'''
    if arg in kwargs:
        if kwargs[arg] is not None:
            print('args1', kwargs[arg])
            return kwargs[arg]
    if arg in defecto:
        if default is None:
            default = arg
        print('args2', defecto[default])
        return defecto[default]
    print(f'No encontró "{arg}" en {kwargs}')
    print('args3')
    return None

def n2s(num,dec=2):
    f="{:."+str(dec)+"f}"
    return f.format(num).rstrip('0').rstrip('.')
    

class Objetos:
    def __init__(self, con=None, **kwargs):
        kwargs['con'] = con
        self.kwargs = kwargs
        print('objetos 1', kwargs)
        self.listado = []
        self.__crea_objetos__(**kwargs)  # objs no devuelve nada
        print('objetos 2', self)
        self.en(xyz(args(kwargs, 'en')))
        nombres=args(kwargs, 'llamad').strip().split()
        for nombre in nombres:
            if nombre in objs:
                objs[nombre] += [self]
            else:
                objs[nombre] = [self]
            
        
        

    def __crea_objetos__(self, **kwargs):
        Objeto_bpy(obj_bpy)

    def dims(self):
        print('No se la dim')
        return None

    def keyframe_insert(self, keyframe='location', frame=0):
        for objeto in self.listado:
            objeto.keyframe_insert(keyframe=keyframe, frame=frame)

    def visible(self, val=True):
        for objeto in self.listado:
            objeto.visible(val)
        return 'hide_viewport', 'hide_render'

    def en(self, loc):
        for objeto in self.listado:
            objeto.en(loc)
        return 'location',
    
    def girado(self, rot):
        for objeto in self.listado:
            objeto.girado(rot)
        return 'rotation',
    


class Objeto:
    def __init__(self, con=None, **kwargs):  
        kwargs['con'] = con
        self.kwargs = kwargs
        de = args(kwargs, 'de') 
        llamad = args(kwargs, 'llamad') 
        print('objeto 1', kwargs)
        obj = self.__crea_objeto__(**kwargs)
        print('objeto 2', obj)
        self.objeto_bpy = obj
        self.en(xyz(args(kwargs, 'en')))
        de_val = 'Material_'+llamad
        mat = bpy.data.materials.get(de_val)
        self.mat = mat
        if mat is None:
            mat = bpy.data.materials.new(de_val)
        if self.objeto_bpy.data.materials:
            self.objeto_bpy.data.materials[0] = mat
        else:
            self.objeto_bpy.data.materials.append(mat)
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = colores[de]
        mat.node_tree.nodes["Principled BSDF"].inputs['Emission'].default_value = colores[de]
        mat.node_tree.nodes["Principled BSDF"].inputs['Emission Strength'].default_value = args(
            kwargs, 'brillo')
        if 'invisible_visibe' in kwargs:
            valor=False
            for frame in kwargs['invisible_visibe']:
                    self.keyframe_insert(self.visible(valor), frame=frame)
                    valor =  not valor
        nombres=args(kwargs, 'llamad').strip().split()
        for nombre in nombres:
            if nombre in objs:
                objs[nombre] += [self]
            else:
                objs[nombre] = [self]        
        if 'gira' in kwargs:
            self.girado(kwargs['gira'])



    def __crea_objeto__(self, **kwargs):
        bpy.ops.mesh.primitive_monkey_add()
        obj_bpy = bpy.context.object
        return obj_bpy

    def dims(self):
        return self.objeto_bpy.dimensions

    def en(self, loc):
        self.objeto_bpy.location = loc
        return 'location',

    def keyframe_insert(self, keyframe='location', frame=0):
        if isinstance(keyframe, str):
            self.objeto_bpy.keyframe_insert(keyframe, frame=frame)
        else:
            for un_keyframe in keyframe:
                self.objeto_bpy.keyframe_insert(un_keyframe, frame=frame)

    def visible(self, val=True):
        self.objeto_bpy.hide_viewport = not val
        self.objeto_bpy.hide_render = not val
        return 'hide_viewport', 'hide_render'
    
    def girado(self, rot):
        self.objeto_bpy.rotation_euler =   rot
        return 'rotation',
    

class Linea(Objeto):
    def __init__(self, con=None, **kwargs):
        super().__init__(con=con, **kwargs)

    def __crea_objeto__(self, **kwargs):
        self.llamad = args(kwargs, 'llamad')
        self.con = args(kwargs, 'con')
        m = bpy.data.meshes.new(self.llamad)
        self.mesh = m
        n = len(self.con)
        m.vertices.add(n)
        m.edges.add(n-1)
        for i, vert in enumerate(self.con):  
            m.vertices[i].co = xyz(vert)
            if i < n-1:
                m.edges[i].vertices = (i, i+1)
        o = bpy.data.objects.new(self.llamad, m)
        o.modifiers.new(type="SKIN", name='Forro')
        bpy.context.collection.objects.link(o)
        for dat in o.data.skin_vertices[''].data:
            dat.radius = (args(self.kwargs, 'ancho_linea'),
                          args(self.kwargs, 'ancho_linea'))
        return o


class Recta_dir_punto(Linea):
    def __init__(self, con=None, **kwargs):
        '''
            con: vector dirección de la recta
        '''
        borde0 = mathutils.Vector(args(kwargs, 'ejes_0'))
        borde1 = mathutils.Vector(args(kwargs, 'ejes_1'))
        try:
            v = mathutils.Vector(con)
        except:
            v = mathutils.Vector(con[0], con[1], 0)
        vertices = []
        if v.x != 0:
            t0 = borde0.x/v.x
            t1 = borde1.x/v.x
            vertices += [t0*v, t1*v]
        if v.y != 0:
            t0 = borde0.y/v.y
            t1 = borde1.y/v.y
            vertices += [t0*v, t1*v]
        if v.z != 0:
            t0 = borde0.z/v.z
            t1 = borde1.z/v.z
            vertices += [t0*v, t1*v]
        vertices_dentro = []
        for vertice in vertices:
            if ((borde0.x <= vertice.x) and
               (vertice.x <= borde1.x) and
               (borde0.y <= vertice.y) and
               (vertice.y <= borde1.y) and
               (borde0.z <= vertice.z) and
               (vertice.z <= borde1.z)):
                vertices_dentro.append(vertice)
        print('vertices_dentro,', vertices_dentro, vertices)
        super().__init__(con=vertices_dentro, **kwargs)

class Flecha(Linea):
    def __init__(self, con=None, **kwargs):
        x,y,z=xyz(con)
        norm=np.sqrt(x**2+y**2+z**2)
        normxy=np.sqrt(x**2+y**2)
        norm1=0.1*norm
        norm9=0.9*norm
        if normxy==0:
            c0=0
            b0= -np.pi/2 if z>0 else +np.pi/2
        else:
            c0=np.arccos(x/normxy)
            c0=c0  if y>0 else 2*np.pi-c0
            b0=np.arccos(normxy/norm)
            b0 = -b0 if z>0 else b0
        a1,b1,c1=kwargs.pop('gira',(0,0,0))
        print('Flecha',x,y,z,':',b0,c0,a1,b1,c1,':',normxy,norm)
        vertices = [[0]*3, (norm,0,0), (norm9,norm1,0), (norm9,-norm1,0),  (norm,0,0),(norm9,0, norm1), (norm9,0,-norm1),  (norm,0,0)]
        super().__init__(con=vertices,gira=(a1,b0+b1,c0+c1), **kwargs)
        
        
#        xyz1 = mathutils.Vector(xyz(con))
#        x1, x2, x3 = xyz1
#        y12 = (0.7*x1-0.7*x2)/10.0
#        y21 = (0.7*x1+0.7*x2)/10.0
#        y23 = (0.7*x2-0.7*x3)/10.0
#        y32 = (0.7*x2+0.7*x3)/10.0
#        u1 = (x1-y12, x2-y21, x3)
#        u2 = (x1-y21, x2+y12, x3)
#        u3 = (x1, x2-y23, x3-y32)
#        xyz_k = xyz1*0.95
#        vertices = [[0]*3, xyz1, u1, xyz_k, u2, xyz1, u3, xyz_k]
#        super().__init__(con=vertices, **kwargs)

class Flechas_nparray(Objetos):
    def __init__(self, con=None, **kwargs):
        # print('Matriz1',kwargs)
        super().__init__(con=con, **kwargs)

    def __crea_objetos__(self, **kwargs):
        con = self.kwargs['con']
        if 'llamad' in kwargs:
            llamad = kwargs['llamad']+'_flechas'
        else:
            llamad = 'Flechas'

        if 'de' in kwargs:
            de = kwargs['de']
            if isinstance(de, str):
                de = [de]
        else:
            de = defecto['algunos_colores']

        for i in range(con.shape[1]):
            print('Flechas_nparray',i)
            i_de = i % len(de)
            kwargs['llamad'] = llamad+'_fle_'+str(i)
            kwargs['de'] = de[i_de]
            kwargs.pop('con', None)
            self.listado.append(Flecha(con=con[:, i], **kwargs), )
        kwargs['llamad'] = llamad

class Texto(Objeto):
    def __init__(self, con=None, **kwargs):
        super().__init__(con=con, **kwargs)

    def __crea_objeto__(self, **kwargs):
        font_curve = bpy.data.curves.new(
            type="FONT", name=args(self.kwargs, 'llamad')+'_Curva')
        font_curve.body = self.kwargs['con']
        font_obj = bpy.data.objects.new(
            name=args(self.kwargs, 'llamad'), object_data=font_curve)
        bpy.context.scene.collection.objects.link(font_obj)
        font_obj.data.size = args(self.kwargs, 'escalado')
        font_obj.data.align_x = 'LEFT'
        font_obj.data.align_y = 'CENTER'
        return font_obj

    def dims(self):
        x, y, z = self.objeto_bpy.dimensions
        return x, y, z

class Punto(Objeto):
    def __init__(self, con=None, **kwargs):
        super().__init__(con=con, **kwargs)

    def __crea_objeto__(self, **kwargs):
        if 'con' in kwargs:
            if kwargs['con']:
                con = kwargs['con']
            else:
                con = defecto['radio_punto']
        else:
            con = defecto['radio_punto']
        kwargs.pop('con', None)
        llamad = args(kwargs, 'llamad')
        mesh = bpy.data.meshes.new(llamad)
        basic_sphere = bpy.data.objects.new(llamad, mesh)
        bpy.context.collection.objects.link(basic_sphere)
        bpy.context.view_layer.objects.active = basic_sphere
        basic_sphere.select_set(True)
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=con)
        bm.to_mesh(mesh)
        bm.free()
        return basic_sphere

class Matriz(Objetos):
    def __init__(self, con=None, **kwargs):
        super().__init__(con=con, **kwargs)

    def __crea_objetos__(self, **kwargs):
        arreglo = np.array(self.kwargs['con'])
        if arreglo.ndim == 1:
            arreglo = arreglo[..., np.newaxis]

        escalado=args(self.kwargs, 'escalado')
        MARGEN = 0.8 *escalado
        ESPACIO_HORIZ = 0.8 *escalado
        llamad = kwargs['llamad']
        objetos2 = []
        for ind_i, objs in enumerate(arreglo):
            objetos1 = []
            for ind_j, j in enumerate(objs):
                kwargs.pop('con', None)
                kwargs['llamad'] = llamad+'_elem_'+str(ind_i)+'_'+str(ind_j)
                if isinstance(j, float):
                    con = "{:.2f}".format(j).rstrip('0').rstrip('.')
                else:
                    con = j
                ob = Texto(con=str(con), **kwargs)
                objetos1.append(ob)
            objetos2.append(objetos1)

        m = len(objetos2)
        n = len(objetos2[0])
        anchos = [0]*n
        altos = [0]*m

        for i in range(m):
            for j in range(n):
                x, y, z = objetos2[i][j].dims()
                anchos[j] = max(anchos[j], x)
                altos[i] = max(altos[i], y)

        alturas = [0]*m
        alturas[0] = -MARGEN/2
        for i in range(1, m):
            alturas[i] = alturas[i-1] - altos[i-1] - MARGEN
        margen_inferior = alturas[-1]-altos[-1] 
        anchuras = [0]*n
        anchuras[0] = MARGEN/2
        for j in range(1, n):
            anchuras[j] = anchuras[j-1] + anchos[j-1]+ESPACIO_HORIZ
        margen_derecha = anchuras[-1]+anchos[-1] + MARGEN/2
        LINEA = altos[0]/2
        self.alturas = alturas
        self.anchuras = anchuras
        self.objetos = objetos2
        kwargs.pop('con', None)
        kwargs['llamad'] = llamad+'_izq'
        self.corchIzq = Linea(con=[(LINEA, altos[0]/2, 0), (0, altos[0]/2, 0),
                              (0, margen_inferior, 0), (LINEA, margen_inferior, 0)], **kwargs)
        kwargs['llamad'] = llamad+'_der'
        self.corchDer = Linea(con=[(margen_derecha-LINEA, altos[0]/2, 0), (margen_derecha, altos[0]/2, 0), (margen_derecha, margen_inferior, 0),
                              (margen_derecha-LINEA, margen_inferior, 0)], **kwargs)
        self.dimens = margen_derecha, 0, margen_inferior
        self.listado = []
        for objs1 in self.objetos:
            self.listado += objs1
        self.listado += [self.corchIzq, self.corchDer]

    def en(self, pos):
        print('Matriz.en', pos)
        dx, dy, dz = self.dims()
        en_x, en_y, en_z = xyz(pos)
        en_y = en_y-dz/2
        objetos2 = self.objetos
        alturas = self.alturas
        anchuras = self.anchuras
        m = len(objetos2)
        n = len(objetos2[0])
        for i in range(m):
            for j in range(n):
                text = objetos2[i][j]
                text.en((anchuras[j]+en_x, alturas[i]+en_y, en_z))
        self.corchIzq.en((en_x, en_y, en_z))
        self.corchDer.en((en_x, en_y, en_z))
        return 'location',

    def dims(self):
        return self.dimens

class Renglon(Objetos):
    def __init__(self, con=None, **kwargs):
        if 'en' not in kwargs:
            kwargs['en'] = (0, global_renglon, 0)
        super().__init__(con=con, **kwargs)

    def __crea_objetos__(self, **kwargs):
        global global_renglon
        self.listado = list(self.kwargs['con'])
        self.pos_objs = []
        pos_x = 0
        pos_y = 0
        pos_z = 0
        pos_z_min = pos_z

        kwargs_txt= kwargs.copy()
    
        llamad=kwargs_txt.pop('llamad','texto')
        kwargs_txt.pop('con',None)
        for i, obj in enumerate(self.listado):
            if isinstance(obj, str):
                obj = Texto(obj, llamad='_txt_'+str(i), **kwargs_txt)
                self.listado[i] = obj
            dim_x, dim_y, dim_z = obj.dims()
            self.pos_objs.append((pos_x, pos_y, pos_z))
            pos_x = pos_x + dim_x + args(kwargs, 'espacio_entre_palabras')
            pos_z_min = min(pos_z_min, dim_z)
            print('dim_z', dim_x, dim_y, dim_z)
        self.dimens = pos_x, pos_y, pos_z  # _min
        global_renglon -= abs(dim_y) + args(kwargs, 'espacio_entre_renglones')

    def en(self, pos):
        for obj, pos_obj in zip(self.listado, self.pos_objs):
            posx, posy, posz = xyz(pos)
            pos_objx, pos_objy, pos_objz = pos_obj
            print('renglon en', (posx+pos_objx, posy+pos_objy, posz+pos_objz))
            obj.en((posx+pos_objx, posy+pos_objy, posz+pos_objz))
        return 'location',

    def dims(self):
        return self.dimens


class Comb_lin(Renglon):
    def __init__(self, A=None, v=None, **kwargs):
        if v.ndim == 1:
            v = v[..., np.newaxis]
        if 'llamad' in kwargs:
            llamad = kwargs['llamad']+'_combLin'
        else:
            llamad = 'combLin'
        lista = []
        for i, esc in enumerate(v[:, 0]):
            lista.append(Texto("{:0.1f}".format(esc), llamad='esc_'+str(i)))
            lista.append(Matriz(A[:, i], llamad='col'+str(i)))
            if i == len(v)-1:
                break
            lista.append('+')
        super().__init__(con=lista, **kwargs)

class Parrafo(Objetos):
    def __init__(self, con=None, **kwargs):
        super().__init__(con=con, **kwargs)

    def __crea_objetos__(self, **kwargs):
        self.listado = self.kwargs['con']
        self.pos_objs = []
        pos_x = 0
        pos_y = 0
        pos_z = 0
        for obj in self.listado:
            dim_x, dim_y, dim_z = obj.dims()
            self.pos_objs.append((pos_x, pos_y, pos_z))
            pos_z = pos_z - dim_z - args(kwargs, 'espacio_entre_renglones')
        self.dimens = pos_x, pos_y, pos_z
        global_renglon += dim_z

    def en(self, pos):
        for obj, pos_obj in zip(self.listado, self.pos_objs):
            posx, posy, posz = pos
            pos_objx, pos_objy, pos_objz = pos_obj
            obj.en((posx+pos_objx, posy+pos_objy, posz+pos_objz))
        return 'location',

    def dims(self):
        return self.dimens

class Poligono(Objeto):
    def __init__(self, con=None, **kwargs):
        super().__init__(con=con, **kwargs)

    def __crea_objeto__(self, **kwargs):
        con = self.kwargs['con']
        llamad = args(kwargs, 'llamad')
        verts = [xyz(vert) for vert in con]
        faces = [range(len(con))]  # [(0,1,2,3)]
        mesh = bpy.data.meshes.new(llamad)
        ob = bpy.data.objects.new(llamad, mesh)
        bpy.context.collection.objects.link(ob)
        mesh.from_pydata(verts, [], faces)
        return ob

class Rectangulo(Poligono):
    def __init__(self, con=None, **kwargs):
        x1, y1, z1 = con
        vertices = [(0, 0, 0), (x1, 0, z1/2), (x1, y1, z1), (0, y1, z1/2)]
        super().__init__(con=vertices, **kwargs)

class Plano_ort_punto(Objetos):
    def __init__(self, con=None, **kwargs):
        '''
        `con`: vector perpendicular al plano
        `con`  y `en` Estan en R3  
        '''
        super().__init__(con=con, **kwargs)

    def __crea_objetos__(self, **kwargs):
        con = args(kwargs, 'con')
        x0, y0, z0 = args(kwargs, 'ejes_0')
        x1, y1, z1 = args(kwargs, 'ejes_1')
        if con[2] != 0:
            def z(x, y):
                a, b, c = con
                return x, y, -(a*x+b*y)/c
            vertices = [z(x0, y0), z(x1, y0), z(x1, y1), z(x0, y1)]
        else:
            vertices_2d = []
            a, b, c = con
            d = 0  # a*px+b*py+c*pz 
            if a != 0:
                x30 = (d-b*y0)/a
                print('x0,x30,x1', x0, x30, x1)
                if x0 <= x30 and x30 <= x1:
                    print('x0,x30,x1', x0, x30, x1)
                    vertices_2d.append((x30, y0))
                x31 = (d-b*y1)/a
                print('x0,x31,x1', x0, x31, x1)
                if x0 <= x31 and x31 <= x1:
                    print('x0,x31,x1', x0, x31, x1)
                    vertices_2d.append((x31, y1))
                print('vert2da', vertices_2d)
            if b != 0:
                y30 = (d-a*x0)/b
                print('y0,y30,y1', y0, y30, y1)
                if y0 <= y30 and y30 <= y1:
                    print('y0,y30,y1', y0, y30, y1)
                    vertices_2d.append((x0, y30))
                y31 = (d-a*x1)/b
                print('y0,y31,y1', y0, y31, y1)
                if y0 <= y31 and y31 <= y1:
                    print('y0,y31,y1', y0, y31, y1)
                    vertices_2d.append((x1, y31))
                print('vert2db', vertices_2d)
            (x4, y4), (x5, y5) = vertices_2d[0:2]
            vertices = [(x4, y4, z0), (x5, y5, z0), (x5, y5, z1), (x4, y4, z1)]

        kwargs.pop('con', None)
        self.listado = [
            Poligono(con=vertices, **kwargs),
            Linea(con=vertices, **kwargs),
        ]


class Plano_abcd(Plano_ort_punto):
    def __init__(self, con=None, **kwargs):
        a, b, c, d = con
        if a != 0:
            P = (d/a, 0, 0)
        elif b != 0:
            P = (0, d/b, 0)
        elif c != 0:
            P = (0, 0, d/c)
        # si no es vertical, despeja z con 0=x y=0
        super().__init__(con=(a, b, c), en=P, **kwargs)

class Plano_abd(Plano_abcd):
    def __init__(self, con=None, **kwargs):
        a, b, d = con
        super().__init__(con=(a, b, 0, d), **kwargs)

class Plano_dirs_punto(Plano_ort_punto):
    def __init__(self, u, v, **kwargs):
        ux, uy, uz = xyz(u)
        vx, vy, vz = xyz(v)
        a = uy*vz-vy*uz
        b = -ux*vz+vx*uz
        c = vy*ux-uy*vx
        # si no es vertical, despeja z con 0=x y=0
        super().__init__(con=(a, b, c), **kwargs)


class Ejes(Objetos):
    def __init__(self, con=None, **kwargs):
        super().__init__(con=con, **kwargs)

    def __crea_objetos__(self, **kwargs):
        con = args(kwargs, 'con')
        kwargs.pop('con', None)

        if 'llamad' in kwargs:
            llamad = kwargs['llamad']+'_eje'
        else:
            llamad = 'Ejes_'
        kwargs.pop('llamad', None)

        ejex0, ejey0, ejez0 = defecto['ejes_0']
        ejex1, ejey1, ejez1 = defecto['ejes_1']
        ejez = 0
        ancho_linea = defecto['ancho_linea']*0.5

        if 'x' in con:
            self.listado.append(Linea(con=[(ejex0, 0, ejez), (ejex1, 0, ejez)],
                                llamad=llamad+'x', de='red', ancho_linea=ancho_linea, **kwargs))
        if 'y' in con:
            self.listado.append(Linea(con=[(0, ejey0, ejez), (0, ejey1, ejez)],
                                llamad=llamad+'y', de='green', ancho_linea=ancho_linea, **kwargs))
        if 'z' in con:
            self.listado.append(Linea(con=[(0, 0, ejez0), (0, 0, ejez1)],
                                llamad=llamad+'z', de='blue', ancho_linea=ancho_linea, **kwargs))
        kwargs['llamad'] = llamad

# class TrMat(Objetos):  # No funciona
#     def __init__(self, con=None, **kwargs):
#         super().__init__(con=con, **kwargs)

#     def __crea_objetos__(self, **kwargs):
#         # con : tiene las dimensdiones del dominio y del codominio
#         if 'con' in kwargs:
#             if kwargs['con']:
#                 con = kwargs['con']
#             else:
#                 con = 3, 3
#         else:
#             con = 3, 3

#         if 'llamad' in kwargs:
#             llamad = kwargs['llamad']+'_TrMat'
#         else:
#             llamad = 'TrMatr_'

#         ejex0, ejey0, ejez0 = defecto['ejes_0']
#         ejex1, ejey1, ejez1 = defecto['ejes_1']
#         centro_dominio = np.array((ejex0-1, 0, 0))
#         centro_codominio = np.array((ejex1+1, 0, 0))
#         m, n = con  # A.shape
#         print('TrMatr', m, n)
#         kwargs.pop('con', None)
#         if n == 2:
#             self.listado.append(
#                 Ejes('xy', en=centro_dominio, llamad='dominio', **kwargs))
#         else:
#             self.listado.append(
#                 Ejes('xyz', en=centro_dominio, llamad='dominio', **kwargs))
#         if m == 2:
#             self.listado.append(
#                 Ejes('xy', en=centro_codominio, llamad='codominio', **kwargs))
#         else:
#             self.listado.append(
#                 Ejes('xyz', en=centro_codominio, llamad='codominio', **kwargs))
#         kwargs['llamad'] = llamad
#         self.dom = centro_dominio
#         self.cod = centro_codominio


if __name__ == '__main__':
    # Lee los nombres de los colores
    with open('/gm2022_0404/Prog_Apli/PWM/colors.json') as json_file:
        colores = json.load(json_file)

    # Borra los objetos anteriores
    for o in bpy.context.scene.objects:
        o.hide_render = False
        o.hide_viewport = False
        if o.type in ('MESH', 'CURVE', 'TEXT', 'SURFACE', 'FONT'):
            o.select_set(True)
        else:
            o.select_set(False)
    bpy.ops.object.delete()
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    bpy.ops.wm.open_mainfile(filepath=bpy.data.filepath)
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'RENDERED'
    bpy.context.scene.eevee.use_bloom = True

    # Variable para ubicar los renglones en el eje z
    global_renglon = 8

    def esp_v(esp=1.5):
        global global_renglon
        global_renglon -= esp  # - defecto['espacio_entre_renglones']

    frame_actual = 1
    frame_siguiente = 2
    def dura(duracion = 1):
        global frame_actual, frame_siguiente
        frame_actual = frame_siguiente
        frame_siguiente = frame_actual + duracion
    
    def obj0(nombre):
        return objs[nombre][0]
    

######################################################################
    # Matriz de rotación
    def mat_rot_grad(a=0):
        a=a/180*np.pi
        return np.array([
            [np.cos(a), -np.sin(a)],
            [np.sin(a),  np.cos(a)]
        ])

#################################################################################################    
#    # Rotacióon en 2D
    #print(mat_rot())
#    Ejes('xy',  llamad='Ejes_2D', invisible_visibe=(0,frame_actual))
#    v=np.array([[1],[0]])
#    
#    A=mat_rot_grad(45)

#    dura(1)
#    Flecha(v,  llamad='brazo1_2D_home', de='yellow', invisible_visibe=(0,frame_actual))
#    dura(1)
#    Renglon((
#        Matriz(A, llamad='rot2x2_A',  invisible_visibe=(0,frame_actual,frame_siguiente)),
#        Matriz(('x','y'), llamad='rot2x2_x',  invisible_visibe=(0,frame_actual,frame_siguiente)),
#        '=',
#        Matriz(((
#            "%.2f" % A[0,0])+'x + '+("%.2f" % A[0,1])+'y'
#            ,("%.2f" % A[1,0])+'x + '+("%.2f" % A[1,1])+'y'), 
#          llamad='rot2x2_Ax',  invisible_visibe=(0,frame_actual,frame_siguiente)),
#    ), llamad='rot2x2', en=(-1, -2, 0),  invisible_visibe=(0,frame_actual,frame_siguiente))
#    Flecha(A@v,  llamad='brazo1_2D_rotado', de='purple', invisible_visibe=(0,frame_actual,frame_siguiente))
#    dura(1)
#    obj0('brazo1_2D_home').keyframe_insert(obj0('brazo1_2D_home').visible(False), frame=frame_actual)
#    obj0('Ejes_2D').keyframe_insert(obj0('Ejes_2D').visible(False), frame=frame_actual)
#    #obj0('Flecha2').keyframe_insert(obj0('Flecha2').visible(False), frame=frame_actual)
#    #dura(1)


##################################################################################################3
###############################################################################################
    #Rotacion en 3D, angulos de Euler

    #Ejes('xyz',  llamad='Ejes', invisible_visibe=(0,frame_actual))

    #A1=mat_rot_grad(60)
    # Rotacióon en el plano xy
    def Rxy(ang):
        A1=mat_rot_grad(ang)
        return np.array([
        [A1[0,0], A1[0,1], 0],
        [A1[1,0], A1[1,1], 0],
        [0,       0,       1],
    ])

    # Rotacióon en el plano xz
    def Rxz(ang):
        A1=mat_rot_grad(ang)
        return np.array([
        [ A1[0,0], 0,  A1[0,1]],
        [ 0,       1,  0      ],
        [ A1[1,0], 0,  A1[1,1]],
    ])


    # Rotacióon en el plano yz
    def Ryz(ang):
        A1=mat_rot_grad(ang)
        return np.array([
        [1, 0,       0      ],
        [0, A1[0,0], A1[0,1]],
        [0, A1[1,0], A1[1,1]],
    ])

    #A=Rxy(60)
    #A=Rxz(80) @ Rxy(10)
    #A=Rxz(60)



    B=np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

#####################################################################################
#Rotacion en 3D, angulos de Euler

#    dura(1)
#    Flechas_nparray(B,  llamad='brazo1_rot_home', de=('red','green','blue'),invisible_visibe=(0,frame_actual))
#    dura(1)
#    #Matriz(A, llamad='A1', en=(-1, -2, 0), invisible_visibe=(0,frame_actual,frame_siguiente))
#    Renglon((
#        Matriz(A, llamad='rot3x3_A',  invisible_visibe=(0,frame_actual,frame_siguiente)),
#        Matriz(('x','y','z'), llamad='rot2x2_x',  invisible_visibe=(0,frame_actual,frame_siguiente)),
#        '=',
#        Matriz((
#            n2s( A[0,0])+'x + '+n2s(A[0,1])+'y + '+n2s(A[0,2])+'z',
#            n2s( A[1,0])+'x + '+n2s(A[1,1])+'y + '+n2s(A[1,2])+'z',
#            n2s( A[2,0])+'x + '+n2s(A[2,1])+'y + '+n2s(A[2,2])+'z',
#            ), 
#          llamad='rot3x3_Ax',  invisible_visibe=(0,frame_actual,frame_siguiente)),
#    ), llamad='rot3x3', en=(-3, -2, 0),  invisible_visibe=(0,frame_actual,frame_siguiente))
#    Flechas_nparray(A@B,   de=('coral','forestgreen','cyan'),llamad='brazo1', invisible_visibe=(0,frame_actual,frame_siguiente))
#    dura(1)
#    obj0('brazo1_rot_home').keyframe_insert(obj0('brazo1_rot_home').visible(False), frame=frame_actual)
#    #obj0('Flecha2').keyframe_insert(obj0('Flecha2').visible(False), frame=frame_actual)


##############################################################################



## Traslación 
#    v=np.array([1,2,3])
#    
#    

#    dura(1)
#    Flechas_nparray(B,  llamad='brazo1_tr_home', de=('red','green','blue'),invisible_visibe=(0,frame_actual))
#    dura(1)
#    #Matriz(A, llamad='A1', en=(-1, -2, 0), invisible_visibe=(0,frame_actual,frame_siguiente))
#    Flecha(v,  llamad='desp', invisible_visibe=(0,frame_actual))
#    dura(1)
#    Renglon((
#        Matriz(v, llamad='rot3x3_A',  invisible_visibe=(0,frame_actual,frame_siguiente)),
#        '+',
#        Matriz(('x','y','z'), llamad='rot2x2_x',  invisible_visibe=(0,frame_actual,frame_siguiente)),
#        '=',
#        Matriz((
#            n2s( v[0])+' + x',
#            n2s( v[1])+' + y',
#            n2s( v[2])+' + z',
#            ), 
#          llamad='rot3x3_Ax',  invisible_visibe=(0,frame_actual,frame_siguiente)),
#    ), llamad='rot3x3', en=(-3, -2, 0),  invisible_visibe=(0,frame_actual,frame_siguiente))
#    
#    Flechas_nparray(B, en=v,   de=('coral','forestgreen','cyan'),llamad='brazo1', invisible_visibe=(0,frame_actual,frame_siguiente))
#    dura(1)
#    obj0('brazo1_tr_home').keyframe_insert(obj0('brazo1_tr_home').visible(False), frame=frame_actual)
#    #obj0('Flecha2').keyframe_insert(obj0('Flecha2').visible(False), frame=frame_actual)

#####################################################################################
#Rotacion en 3D  y luego una traslación 
#    v=np.array([1,2,3])
#    A=Rxz(45)# @ Rxy(10)

#    dura(1)
#    Flechas_nparray(B,  llamad='brazo1_rot_home', de=('red','green','blue'),invisible_visibe=(0,frame_actual))
#    dura(1)
#    #Matriz(A, llamad='A1', en=(-1, -2, 0), invisible_visibe=(0,frame_actual,frame_siguiente))
#    Renglon((
#        Matriz(A, llamad='rot3x3_A',  invisible_visibe=(0,frame_actual,frame_siguiente)),
#        Matriz(('x','y','z'), llamad='rot2x2_x',  invisible_visibe=(0,frame_actual,frame_siguiente)),
#        '+',
#        Matriz(v, llamad='trasl_v',  invisible_visibe=(0,frame_actual,frame_siguiente)),
#        '=',
#        Matriz((
#            n2s( A[0,0])+'x + '+n2s(A[0,1])+'y + '+n2s(A[0,2])+'z +'+n2s(v[0]),
#            n2s( A[1,0])+'x + '+n2s(A[1,1])+'y + '+n2s(A[1,2])+'z +'+n2s(v[1]),
#            n2s( A[2,0])+'x + '+n2s(A[2,1])+'y + '+n2s(A[2,2])+'z +'+n2s(v[2]),
#            ), 
#          llamad='rot3x3_Ax',  invisible_visibe=(0,frame_actual,frame_siguiente)),
#    ), llamad='rot3x3', en=(-3, -2, 0),  invisible_visibe=(0,frame_actual,frame_siguiente))
#    Flecha(v,  llamad='tra', invisible_visibe=(0,frame_actual,frame_siguiente))
#    Flechas_nparray(A@B,  en=v, de=('coral','forestgreen','cyan'),llamad='brazo1', invisible_visibe=(0,frame_actual,frame_siguiente))
#    dura(1)
#    obj0('brazo1_rot_home').keyframe_insert(obj0('brazo1_rot_home').visible(False), frame=frame_actual)
#    #obj0('Flecha2').keyframe_insert(obj0('Flecha2').visible(False), frame=frame_actual)





##############################################################################



#Brazos 

    primer_angulo = Rxz(-50)
    primer_brazo=np.array([1,0,0])
    segundo_angulo = Ryz(-50) @ primer_angulo
    segundo_brazo=np.array([1,0,0])
    tercer_angulo = Rxy(-40) @ segundo_angulo
    tercer_brazo=np.array([1,0,0])
    
    
    primer_brazo_gir  = primer_angulo @ primer_brazo 
    segundo_brazo_gir = segundo_angulo @ segundo_brazo
    tercer_brazo_gir  = tercer_angulo  @ tercer_brazo
    
    primer_codo=primer_brazo_gir
    segundo_codo=segundo_brazo_gir + primer_codo
    tercer_codo=tercer_brazo_gir + segundo_codo 
    
    Flecha(primer_brazo_gir, en=(0,0,0),  llamad='primer_brazo',invisible_visibe=(0,frame_actual))
    Flecha(segundo_brazo_gir, en=primer_codo,  llamad='segundo_brazo',invisible_visibe=(0,frame_actual))
    Flecha(tercer_brazo_gir, en=segundo_codo,  llamad='tercer_brazo',invisible_visibe=(0,frame_actual))

    print('tercer_codo',tercer_codo)

    if ( (1<tercer_codo[0] and tercer_codo[0]>3)
        and (-1<tercer_codo[1] and tercer_codo[1]>1)
        and (0<tercer_codo[2] and tercer_codo[2]>5) ) :
        print ('Está en la zona 1') 














