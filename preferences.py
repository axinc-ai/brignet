import os
import sys
import bpy


class BrignetPrefs(bpy.types.AddonPreferences):
    bl_idname = __package__

    @staticmethod
    def append_modules():
        env_path = bpy.context.preferences.addons[__package__].preferences.modules_path

        if not os.path.isdir(env_path):
            return False

        if sys.platform.startswith("linux"):
            lib_path = os.path.join(env_path, 'lib')
            sitepackages = os.path.join(lib_path, 'python3.7', 'site-packages')
        else:
            lib_path = os.path.join(env_path, 'Lib')
            sitepackages = os.path.join(lib_path, 'site-packages')

        if not os.path.isdir(sitepackages):
            # not a python path, but the user might be still typing
            return False

        platformpath = os.path.join(sitepackages, sys.platform)
        platformlibs = os.path.join(platformpath, 'lib')

        mod_paths = [lib_path, sitepackages, platformpath, platformlibs]
        if sys.platform.startswith("win"):
            mod_paths.append(os.path.join(env_path, 'DLLs'))
            mod_paths.append(os.path.join(sitepackages, 'Pythonwin'))

        for mod_path in mod_paths:
            if not os.path.isdir(mod_path):
                # TODO: warning
                continue
            if mod_path not in sys.path:
                sys.path.append(mod_path)

        sys.path.append(env_path)
        return True

    @staticmethod
    def append_rignet():
        rignet_path = bpy.context.preferences.addons[__package__].preferences.rignet_path

        if not os.path.isdir(rignet_path):
            return False

        dirname = os.path.dirname(rignet_path) if rignet_path.endswith(os.sep) else rignet_path
        dirname = os.path.dirname(dirname)
        dirname = os.path.dirname(dirname)
        util_dir = os.path.join(dirname, "util")
        if not os.path.isdir(util_dir):
            # not the rignet path, but the user might still be typing
            return False

        if not rignet_path in sys.path:
            sys.path.append(rignet_path)
        if not util_dir in sys.path:
            sys.path.append(util_dir)

        return True

    def update_modules(self, context):
        self.append_modules()

    def update_rignet(self, context):
        self.append_rignet()

    modules_path: bpy.props.StringProperty(
        name='RigNet environment path',
        description='Path to Conda RignetEnvironment',
        subtype='DIR_PATH',
        update=update_modules
    )

    rignet_path: bpy.props.StringProperty(
        name='ailia-rignet path',
        description='Path to ailia-rignet',
        subtype='DIR_PATH',
        default=os.path.join('', 'ailia-models', 'rigging', 'rignet'),
        update=update_rignet
    )

    def draw(self, context):
        layout = self.layout
        column = layout.column()
        box = column.box()

        # first stage
        col = box.column()
        row = col.row()
        row.prop(self, 'modules_path', text='Modules Path')
        row = col.row()
        row.prop(self, 'rignet_path', text='ailia-rignet Path')

        row = layout.row()
        row.label(text="End of bRigNet Preferences")
