from OpenGL.GL import GL_DEPTH_WRITEMASK, GL_FALSE, GL_TRUE, glDepthMask, glGetBooleanv


def get_depth_mask() -> bool:
    value = glGetBooleanv(GL_DEPTH_WRITEMASK)
    try:
        return bool(value[0])
    except (TypeError, IndexError):
        return bool(value)


def set_depth_mask(enabled: bool) -> None:
    glDepthMask(GL_TRUE if enabled else GL_FALSE)
