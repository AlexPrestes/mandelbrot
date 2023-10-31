import numpy as np
import taichi as ti
from matplotlib import colormaps

ti.init(arch=ti.cuda)


@ti.data_oriented
class BigArexFloat:
    def __init__(self) -> None:
        pass

@ti.data_oriented
class Fractal:
    def __init__(self, width=1500, height=1000) -> None:
        self.width = width
        self.height = height
        self.res = (width, height)
        self.frame = ti.Vector.field(3, ti.f32, shape=self.res)
        self.window = ti.ui.Window('pygame', self.res, pos = ((1920-width)//2, (1080-height)//2))
        self.canvas = self.window.get_canvas()
        self.xscale = ti.Vector([-2, 1], ti.f64)
        self.yscale = ti.Vector([-1, 1], ti.f64)
        self.size_pixel = (self.xscale.y-self.xscale.x)/self.width
        self.offset = ti.Vector( [-0.5, 0.0], ti.f64)
        self.max_iter_factor = 32
        self.max_iter = -int(np.log10(self.size_pixel)*self.max_iter_factor)
        self.paused_zoom = True
        self.cmap = ti.Vector.field(3, ti.f64, shape=(256,))
        self.cmap.from_numpy(colormaps['prism'](np.linspace(0, 1, 256))[:, :3])
    
    def events(self):
        if self.window.get_event( ti.ui.PRESS ):
            if self.window.event.key in ['q', 'Q']:
                exit()

            if self.window.event.key in ['r', 'R', ti.ui.RMB]:
                self.offset = ti.Vector( [-0.5, 0.0], ti.f64)
                self.xscale = ti.Vector([-2, 1])
                self.yscale = ti.Vector([-1, 1])
                self.size_pixel = (self.xscale.y-self.xscale.x)/self.width
                self.max_iter = -int(np.log10(self.size_pixel)*self.max_iter_factor)

            if self.window.event.key == ti.ui.SPACE:
                self.paused_zoom = not self.paused_zoom
            
            if self.window.event.key == ti.ui.LMB:
                self.adjust_offset()
            
            if self.window.event.key == ti.ui.MMB:
                print(self.offset)

    
    def adjust_offset(self):
        p = ti.Vector(self.window.get_cursor_pos(), ti.f64)
        x = self.offset.x +(p[0]-0.5)*self.width*self.size_pixel
        y = self.offset.y +(p[1]-0.5)*self.height*self.size_pixel
        self.offset = ti.Vector([x, y])

    def adjust_zoom(self, factor=5):
        x = self.xscale.x +3*factor*self.size_pixel
        y = self.xscale.y -3*factor*self.size_pixel
        self.xscale -= self.xscale-ti.Vector([x, y])

        x = self.yscale.x +2*factor*self.size_pixel
        y = self.yscale.y -2*factor*self.size_pixel
        self.yscale -= self.yscale-ti.Vector([x, y])

        self.size_pixel = (self.xscale.y-self.xscale.x)/self.width
        self.max_iter = -int(np.log10(self.size_pixel)*self.max_iter_factor)

    @ti.func
    def get_color(self, val: int):
        c = self.cmap[val]
        return c

    @ti.func
    def calc_z(self, x: ti.u64, y: ti.u64, size_pixel: ti.f64, offset: ti.math.vec2, max_iter: ti.u64):
        cx = offset.x +size_pixel*(x-self.width//2)
        cy = offset.y +size_pixel*(y-self.height//2)
        c = ti.Vector([cx, cy], ti.f64)
        z = ti.Vector([0 , 0 ], ti.f64)
        i = 0
        zz = z.x**2 +z.y**2
        while i < max_iter and zz < 4:
            z = ti.Vector([ (z.x**2 -z.y**2 +c.x), (2*z.x*z.y +c.y) ])
            zz = z.x**2 +z.y**2
            i += 1
        
        col_i = int(255*i/max_iter)
        if zz < 4:
            col_i = int(0)
        
        color = self.get_color(col_i)
        return color
    
    @ti.kernel
    def update(self, size_pixel: ti.f64, offset: ti.math.vec2, max_iter: ti.u64):
        for x, y in self.frame:
            self.frame[x, y] = self.calc_z(x, y, size_pixel, offset, max_iter)
    
    def run(self):
        while self.window.running:
            #self.offset = ti.Vector([-0.77568377, 0.13646737])
            self.events()
            if not self.paused_zoom:
                self.adjust_zoom(2)
            self.update(self.size_pixel, self.offset, self.max_iter)
            self.canvas.set_image(self.frame)
            self.window.show()


if __name__ == '__main__':
    app = Fractal()
    app.run()