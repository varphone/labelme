from __future__ import annotations

from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from .._shape import Shape
from .canvas import Canvas


class MinimapWidget(QtWidgets.QFrame):
    """A small overview of the image and the canvas' current viewport."""

    center_requested = QtCore.Signal(QtCore.QPointF)

    _MARGIN = 10
    _WIDTH = 180
    _HEIGHT = 120

    def __init__(
        self, *, canvas: Canvas, viewport: QtWidgets.QWidget
    ) -> None:
        super().__init__(parent=viewport)
        self.canvas = canvas
        self.viewport = viewport
        self.setFixedSize(self._WIDTH, self._HEIGHT)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        viewport.installEventFilter(self)
        self._dragging = False
        self._show_shape_outlines = True
        self._position_in_viewport()

    def set_show_shape_outlines(self, value: bool) -> None:
        self._show_shape_outlines = value
        self.update()

    def set_mouse_passthrough(self, value: bool) -> None:
        """Let drawing gestures reach the canvas beneath the overview."""
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, value)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.viewport and event.type() == QtCore.QEvent.Type.Resize:
            self._position_in_viewport()
        return super().eventFilter(watched, event)

    def _position_in_viewport(self) -> None:
        if (
            self.viewport.width() < self.width() + 2 * self._MARGIN
            or self.viewport.height() < self.height() + 2 * self._MARGIN
        ):
            # Do not cover the canvas when the viewport is narrower than the
            # overview itself (for example while the dock layout is compact).
            self.move(-self.width() - 1, -self.height() - 1)
            return
        self.move(
            max(self.viewport.width() - self.width() - self._MARGIN, 0),
            self._MARGIN,
        )
        self.raise_()

    def _image_rect(self) -> QtCore.QRectF:
        pixmap = self.canvas.pixmap
        return QtCore.QRectF(0, 0, pixmap.width(), pixmap.height())

    def _map_rect(self) -> QtCore.QRectF:
        image_rect = self._image_rect()
        if image_rect.isEmpty():
            return QtCore.QRectF()
        margin = 8
        target = QtCore.QRectF(
            margin,
            margin,
            self.width() - 2 * margin,
            self.height() - 2 * margin,
        )
        scale = min(
            target.width() / image_rect.width(),
            target.height() / image_rect.height(),
        )
        size = QtCore.QSizeF(image_rect.size()) * scale
        return QtCore.QRectF(
            target.center() - QtCore.QPointF(size.width() / 2, size.height() / 2),
            size,
        )

    def _image_to_minimap(self, point: QtCore.QPointF) -> QtCore.QPointF:
        image_rect = self._image_rect()
        map_rect = self._map_rect()
        if image_rect.isEmpty() or map_rect.isEmpty():
            return QtCore.QPointF()
        return map_rect.topLeft() + QtCore.QPointF(
            point.x() / image_rect.width() * map_rect.width(),
            point.y() / image_rect.height() * map_rect.height(),
        )

    def _minimap_to_image(self, point: QtCore.QPointF) -> QtCore.QPointF:
        image_rect = self._image_rect()
        map_rect = self._map_rect()
        if image_rect.isEmpty() or map_rect.isEmpty():
            return QtCore.QPointF()
        x = (point.x() - map_rect.left()) / map_rect.width()
        y = (point.y() - map_rect.top()) / map_rect.height()
        return QtCore.QPointF(
            max(0.0, min(image_rect.width(), x * image_rect.width())),
            max(0.0, min(image_rect.height(), y * image_rect.height())),
        )

    def _viewport_image_rect(self) -> QtCore.QRectF:
        top_left = QtCore.QPointF(
            self.canvas.mapFrom(self.viewport, QtCore.QPoint(0, 0))
        )
        bottom_right = QtCore.QPointF(
            self.canvas.mapFrom(self.viewport, self.viewport.rect().bottomRight())
        )
        return self.canvas.viewport_image_rect(
            top_left=top_left, bottom_right=bottom_right
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        pixmap = self.canvas.pixmap
        map_rect = self._map_rect()
        if pixmap.isNull() or map_rect.isEmpty():
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(map_rect, pixmap, QtCore.QRectF(pixmap.rect()))

        if self._show_shape_outlines:
            shape_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 220))
            shape_pen.setWidth(1)
            painter.setPen(shape_pen)
            for shape in self.canvas.shapes:
                self._paint_shape(painter=painter, shape=shape)

        view = self._viewport_image_rect()
        view_top_left = self._image_to_minimap(view.topLeft())
        view_bottom_right = self._image_to_minimap(view.bottomRight())
        view_rect = QtCore.QRectF(view_top_left, view_bottom_right).normalized()
        view_pen = QtGui.QPen(QtGui.QColor(255, 80, 80, 240))
        view_pen.setWidth(2)
        painter.setPen(view_pen)
        painter.setBrush(QtGui.QColor(255, 80, 80, 45))
        painter.drawRect(view_rect)
        painter.end()

    def _paint_shape(self, *, painter: QtGui.QPainter, shape: Shape) -> None:
        if len(shape.points) == 0:
            return
        path = QtGui.QPainterPath()
        path.moveTo(self._image_to_minimap(QtCore.QPointF(*shape.points[0])))
        for point in shape.points[1:]:
            path.lineTo(self._image_to_minimap(QtCore.QPointF(*point)))
        if shape.closed:
            path.closeSubpath()
        painter.drawPath(path)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._map_rect().isEmpty():
                event.accept()
                return
            self._dragging = True
            self.center_requested.emit(self._minimap_to_image(event.position()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging:
            if self._map_rect().isEmpty():
                event.accept()
                return
            self.center_requested.emit(self._minimap_to_image(event.position()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)
