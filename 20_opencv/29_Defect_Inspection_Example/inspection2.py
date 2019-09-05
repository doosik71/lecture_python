import cv2
import kivy.app
import kivy.clock
import kivy.graphics.texture
import kivy.uix.image
import kivy.uix.boxlayout
import numpy as np

kivy.require('1.11.1')


def get_diff_image(background, foreground):
    if background is None:
        return foreground
    elif foreground is None:
        return background
    else:
        return np.absolute(background.astype(np.int)
                           - foreground.astype(np.int)).astype(np.uint8)


class InspectionWindow(kivy.uix.boxlayout.BoxLayout):
    def __init__(self, **kwargs):
        super(InspectionWindow, self).__init__(**kwargs)

        self.live_image = None
        self.normal_image = None
        self.defect_image = None
        self.normal_vector = None
        self.defect_vector = None
        self.normal_length = None
        self.defect_length = None

        # 카메라를 연다.
        self.capture = cv2.VideoCapture(0)

        # 주기적으로 화면 갱신을 요청한다.
        kivy.clock.Clock.schedule_interval(self.update_live_image, 0.1)

    @staticmethod
    def image_to_texture(frame):
        # 영상을 상하 반전하고, 문자열 버퍼로 변환한다.
        image_buffer = cv2.flip(frame, 0).tostring()

        # Kivy 텍스처로 변환한다.
        texture = kivy.graphics.texture.Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(image_buffer, colorfmt='bgr', bufferfmt='ubyte')

        return texture

    def capture_image(self):
        # 카메라 영상을 얻는다.
        success, frame = self.capture.read()

        # 영상 획득에 성공하면,
        if success:
            return frame, InspectionWindow.image_to_texture(frame)
        else:
            return None, None

    def update_live_image(self, dt):
        self.live_image, self.ids.live_image.texture = self.capture_image()

        self.calculate_distance()

    def register_normal_image(self):
        self.normal_image , self.ids.normal_image.texture = self.capture_image()
        self.normal_vector = self.normal_image.reshape(-1).astype(int)
        self.normal_length = np.linalg.norm(self.normal_vector)

    def register_defect_image(self):
        self.defect_image, self.ids.defect_image.texture = self.capture_image()
        self.defect_vector = self.defect_image.reshape(-1).astype(int)
        self.defect_length = np.linalg.norm(self.defect_vector)

    def calculate_distance(self):
        if self.normal_image is None:
            self.ids.result.text = 'Register normal image!'
        elif self.defect_image is None:
            self.ids.result.text = 'Register defect image!'
        else:
            target_vector = (self.live_image).reshape(-1).astype(int)
            target_length = np.linalg.norm(target_vector)

            normal_score = np.dot(self.normal_vector, target_vector) / (self.normal_length * target_length)
            defect_score = np.dot(self.defect_vector, target_vector) / (self.defect_length * target_length)

            if normal_score > defect_score:
                decision = 'Normal'
            elif normal_score < defect_score:
                decision = 'Defect'
            else:
                decision = 'Equal'

            self.ids.result.text = '{} ({} vs. {})'.format(decision,
                                                            round(normal_score, 3),
                                                            round(defect_score, 3))

    def close(self):
        if self.capture is not None:
            self.capture.release()

        exit()


class Inspection2App(kivy.app.App):
    def build(self):
        return InspectionWindow()


if __name__ == '__main__':

    Inspection2App().run()

