import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, BatchNormalization, Activation, Dropout, Dense, AveragePooling1D
from tensorflow.keras.models import Model

# 1. Legendre basis: 논문 식(6)에서 제시한 basis function 생성함수
def legendre_basis_functions(q, length):
    # 입력 신호의 domain [-1, 1]을 discretize
    x = np.linspace(-1, 1, length)
    '''
    np.one_like(x): Legendre 0차 다항식
    x: Legendre 1차 다항식
    0.5 * (3 * x**2 - 1): Legendre 2차 다항식
    0.5 * (5 * x**3 - 3 * x): Legendre 3차 다항식
    (1/8) * (35 * x**4 - 30 * x**2 + 3): Legendre 4차 다항식
    '''
    basis = [
        np.ones_like(x),
        x,
        0.5 * (3 * x**2 - 1),
        0.5 * (5 * x**3 - 3 * x),
        (1/8) * (35 * x**4 - 30 * x**2 + 3),
    ]
    # q개의 basis 함수 shape [q, length]로 반환
    return np.stack(basis[:q], axis=0).astype(np.float32)

# 2. BasisProjection Layer
class BasisProjection(Layer):
    def __init__(self, basis_functions, **kwargs):
        super().__init__(**kwargs)
        #  # [q, T] shape, 논문 식(6) φ_i(t), basis function 값 테이블
        self.basis = tf.constant(basis_functions)
        # basis function의 개수 (q), 논문에서 basis 확장 차수
        self.q = self.basis.shape[0]
        # 시간축 길이, 즉 한 basis function의 길이
        self.T = self.basis.shape[1]

    def call(self, inputs): 
        # inputs: [batch size, time, channel]을 basis projection을 위해 [batch size, channel, time]으로 변환
        x = tf.transpose(inputs, [0, 2, 1]) 
        '''
        basis projection
        - 각 배치, 각 채널별로 q개 basis function(φ_i(t))에 projection(내적)을 함
        - tf.einsum('bct,qt->bcq', x, self.basis)
            · x: [batch size, channel, time], self.basis: [q, time]
            · 결과: [batch size, channel, q] (각 채널 신호를 basis function φ_i(t)에 대해 내적(적분)한 값 → basis coefficient)
        - / self.T: 적분의 평균치(정규화), 수치적분 근사
        '''
        proj = tf.einsum('bct,qt->bcq', x, self.basis) / self.T

        '''
        return: [batch size, channel, q]
        - 각 배치, 각 채널별로 basis coefficient(q개)를 반환
        - 이 값이 바로 논문에서 말하는 "basis function 선형결합 계수"
        '''
        return proj

# 3. FunctionalConv1D Layer
class FunctionalConv1D(Layer):
    def __init__(self, filters, q, **kwargs):
        super().__init__(**kwargs)
        # 출력 filter 개수
        self.filters = filters
        # basis function 개수, 논문의 q
        self.q = q

    def build(self, input_shape):  # [batch size, channel, q]
        self.in_channels = input_shape[1]

        # basis coefficient 방향별로 trainng parameter을 정의
        self.weight = self.add_weight(
            shape=(self.in_channels, self.filters, self.q),
            initializer='glorot_uniform',
            trainable=True,
            name='basis_weights'
        )

        # 각 filter별 bias
        self.bias = self.add_weight(
            shape=(self.filters,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )

    def call(self, inputs):
        '''
        inputs: [batch, channel, q]
        weights: [channel, filter, q]

        einsum('bcq,cfq->bf):
            각 배치마다, 모든 입력 channel과 q 방향에 대해 입력 basis coefficient x filter basis coefficient 곱의 합산
            입력 basis coefficient × 필터 basis coefficient 곱의 합산 (즉, basis 차원 내적과 채널 합산)
            논문에서 functional convolution을 필터 basis 선형결합 및 basis coefficient 내적으로 근사하는 연산

        '''
        return tf.einsum('bcq,cfq->bf', inputs, self.weight) + self.bias  # [B, filters]

# 4. FunctionalDense Layer
class FunctionalDense(Layer):
    def __init__(self, num_classes, input_dim, q, **kwargs):
        super().__init__(**kwargs)
        # 출력 class 개수
        self.num_classes = num_classes
        # 입력 ckdnjs (basis projection 이후 channel 수)
        self.input_dim = input_dim
        # basis function의 개수
        self.q = q

    def build(self, input_shape):
        # 각 class별, 입력 channel별, basis별 weights parameter
        self.class_basis_weights = self.add_weight(
            shape=(self.num_classes, self.input_dim, self.q),
            initializer='glorot_uniform',
            trainable=True,
            name='class_basis_weights'
        )
        # 각 class별 basis
        self.bias = self.add_weight(
            shape=(self.num_classes,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )

    def call(self, inputs):
        '''
        input: [batch, channel, q]
        class_basis_weights: [class, channel, q]
        
        enisum('bcq, ncq->bn'):
            각 batch, 각 class에 대해 입력의 basis coefficient의 weights basis coefficient 내적
            논문 수식의 적분 + 합산에 해당 (basis 공간에서의 곱셈/합산산)
        '''
        logits = tf.einsum('bcq,ncq->bn', inputs, self.class_basis_weights) + self.bias
        return tf.nn.softmax(logits)

# 5. 전체 FNN(40) 모델 빌더
def build_FNN40_basis(input_shape=(250, 22), num_classes=8, q=5):
    # 논문 식(6) basis 생성 (q=5, 입력 길이=250)
    basis = legendre_basis_functions(q, length=input_shape[0])
    # [time, channel] → [250, 22]
    inputs = Input(shape=input_shape) 

    # Step 1: projection
    x_proj = BasisProjection(basis)(inputs)  # [batch size, 22, 5]

    # Step 2: FunctionalConv1D (22 → 40): 22 channel, 5 basis coefficient를 40개의 functional filter에 매핑
    x = FunctionalConv1D(filters=40, q=q)(x_proj)  # [batch size, 40]
    x = Activation('elu')(x)

    # Step 3: FunctionalConv1D (40 → 40)
    x = tf.expand_dims(x, axis=1)       # [batch size, 1, 40]
    x = tf.tile(x, [1, q, 1])           # [batch size, 5, 40]
    x = tf.transpose(x, [0, 2, 1])      # [batch size, 40, 5]
    # 40 functional channel의 basis coefficient들을 다시 40 functional filter에 매핑
    x = FunctionalConv1D(filters=40, q=q)(x)  # [batch size, 40]
    x = Activation('elu')(x)

    # Step 4: Dense(40 → 110) → reshape to [batch size, 22, 5]
    x = Dense(22 * q)(x)  # 40 → 110
    x = tf.reshape(x, [-1, 22, q])  # [batch size, 22, 5]

    # Step 5: FunctionalDense: 각 batch 샘플마다, 
    # 22 channel × 5 basis coefficient를 각 class 별 basis weight와 내적하여 softmax로 class 확률 출력
    outputs = FunctionalDense(num_classes=num_classes, input_dim=22, q=q)(x)

    model = Model(inputs, outputs)
    return model
