# Deep-Generative-Replay-review
**논문 읽기 전 사전 설명**
### 평생 학습이란 ?
* 인간의 뇌는 배경지식을 바탕으로 새로운 것을 배우며, 과거의 지식을 잊지 않는다.
* **Lifelong learning**은 인간의 인지를 모방하여 **Catastrophic forgetting** 문제와 **Semantic drift**를 해결하기 위해 등장하였다.
* 그 중 memory 접근법은 생물학적 기억 메커니즘을 모방하자는 아이디어에서 출발했으며, 대표적으로 DGR이 있다.
* 즉, 과거의 **episodic memory system**에 의존하지만, 모두 기억하기에는 너무 큰 메모리가 소요된다는 단점이 있으며, 이에 대한 대안으로 나온 것이 **DGR**

### DGR (Deep Generative Reply) 이란 ?
* 뇌의 해마를 모방하여 만든 알고리즘
* 해마는 감각정보를 단기간 저장하고 있다가, 대뇌피질로 보내 장기기억으로 저장 or 삭제함. 이런 단기기억 & 장기기억의 상보적 학습 관계를 **generator** 와 **solver**로 구현하였다.
* 과거의 데이터를 저장하지 않으며, 대신 generate된 pseudo-data를 동시에 replay한다.
* 즉, 과거의 데이터를 따라하는 pseudo-data를 만들기 위해 generator가 사용된다.
* Generator는 GAN을 기반으로 함. GAN은 학습했던 데이터와 유사한 데이터를 재현함으로써 해마다 단기기억을 저장하는 것과 같은 역할을 한다.
* Solver는 주어진 task를 해결하는 장기기억의 역할. 새로운 task를 학습할 때 Generator가 생성한 이전 task에 대한 데이터를 동시에 학습한다.
* Task A와 Task B의 데이터를 모두 학습하는 것과 같은 효과 발생하여 모델이 Multi task를 수행하도록 한다.
----
### Continual Learning with Deep Generative Replay 리뷰
#### 2.2 Deep Generative Models
* GAN framework는 **generator G** 와 **discriminator D** 사이에 일종의 **zero-sum game**을 한다. D 가 두 데이터 분포를 비교하여 생성된 샘플과 실제 샘플을 구별하는 방법을 배우는 동안, G 는 가능한 실제 분포를 모방하는 방법을 학습한다.
#### 3. Generative Replay
* 먼저 용어를 정의할 것, 우리의 연속적인 학습 프레임워크에서 우리는 N개의 tasks들의 task 순서 T = ($T_1, T_2,...,T_n)$로 풀어야 할 task 순서를 정의한다.
* **Definition 1.** Task Ti 는 학습 예시 ($x_i,y_i$) 를 추출한 데이터 분포 Di의  objective 를 향해 모델을 최적화 하는 것이다.
* 다음으로 우리의 model은 새로운 task를 배우고 그 지식을 다른 네트워크에 학습 시킬 수 있기 때문에 우리는 model을 **scholar** 라고 부를 것이다.
* **Definition 2.** scholar H는 튜플 (G Si)로 이루어져 있음. 
G는 실제와 같은 샘플을 생성하는 generative model 이고, S는 $\theta$ 에 의해 매개변수화된 task solving model 이다.
* Solver는 task sequence T의 모든 task를 수행해야 한다. 전체 목표는 task sequence 의 모든 task간 편향되지 않는 loss들의 합계를 최소화 하는 것.
* 여기서 D는 전체 데이터 분포이고 L는 손실함수이다.
* Ti task를 위해 학습하는 동안에도 모델은 Di에서 추출한 샘플을 제공 받는다.
* 쉽게 말하면, **generator는 replayed된 input data를 생성**하고 , **solver는 real data + generated input 두 종류의 데이터를 사용**하여 모델을 학습, 이런 **generator와 solver를 합친 것을 scholar model** 이라 부른다.
#### 3.1 Proposed Method
* scholar model에 대한 순차적 교육을 고려할 것. single scholar model을 가장 최근의 copy of the network을 참조하여 학습하는 것은 N번째 scholar가 현재 task와 이전 scholar의 knowledge 를 학습하는 과정을 의미하는 sequence of scholar models 학습하는 것과 같음
  
<img width="700" alt="스크린샷 2023-07-21 오후 2 05 25" src="https://github.com/ahhnchangjun/Deep-Generative-Replay-review/assets/125349194/afd77310-0d41-4cb2-9821-236bf36d5253">

* 다른 scholar로부터 온 scholar 모델을 학습하는 것은 generator와 solver를 학습시키는 두가지의 독립적인 절차를 거친다.
* 첫째, 새로운 generator는 현재 task의 input x와 이전 task로부터 replay된 input x을 전송받는다.
* Real sample 과 replayed sample은 이전 task에 비해 새 task에서 요구하는 중요도에 따른 비율로 혼합된다.
* Generator는 cumulative input space를 재구성하는 방법을 학습하고, 새 solver는 실제 replay된 데이터와 동일한 혼합에서 나온 타켓을 input에 결합하도록 학습된다.
* 여기서 replay된 target은 replay된 input에 대한 solver의 응답을 지나친다. 아래는 i 번째 solver에 대한 loss function이다.
  
  <img width="648" alt="스크린샷 2023-07-21 오후 2 12 14" src="https://github.com/ahhnchangjun/Deep-Generative-Replay-review/assets/125349194/737ff854-56f8-4bff-a6ce-682a67a58aba">
  
* $\theta_i$는 i 번째 scholar의 네트워크 parameter이고, r은 mixing 된 real data의 비율이다. 오른쪽은 original task에서 model을 평가하는 것을 목표로 하기 때문에, test loss는 조금 다르며 아래와 같다.
  
<img width="577" alt="스크린샷 2023-07-21 오후 2 12 39" src="https://github.com/ahhnchangjun/Deep-Generative-Replay-review/assets/125349194/d51c4f3f-2f84-4e50-b3fb-8a08a1e97233">

* 여기서 $D_{past}^{}$는 과거 데이터의 누적 분포, i = 1 일때 첫 번째 solver에 대해 참조할 재생 데이터가 없고 두 기능 모두 두 번째 loss항이 무시된다.
* 우리는 scholar model을 task sequence 를 해결하는데에 적합한 architecture을 가진 solver와 GAN framework에서 학습된 generator을 사용하여 scholar 모델을 구축했으며, 우리의 framework는 모든 deep generative model도 generator로 사용할 수 있다.

#### 3.2 Preliminary Experiment
* 학습된 scholar 모델만으로도 비어 있는 network를 훈련시키기에 충분하다는 걸 보여주기 위한 실험. MNIST 숫자 DB 분류에 대한 모델을 테스트 했다.
* Scholar 모델 sequence는 이전 scholar로 부터 generative replay를 통해 처음부터 훈련되었다.
  
  <img width="685" alt="image" src="https://github.com/ahhnchangjun/Deep-Generative-Replay-review/assets/125349194/806d4eb8-c9f5-4434-bbf4-bc124e05bf81">
* 첫 번째 solver는 실제 데이터로부터 배우고, 후속 solver는 이전 scholar 네트워크에서 G, R을 통해 처음부터 훈련되었다.
* scholar 모델이 정보를 잃지 않고 지식으로 전달하는 것을 관찰한다.
#### 4. Experiments
* 본 섹션에서는 다양한 Lifelong learning 설정에 대한 generative replay framework의 적용 가능성을 보여준다.
* 훈련된 scholar network를 기반으로한 Generative replay은 생성 모델의 품질이 작업 성능의 유일한 제약 조건이라는 점에서 다른 연속 학습 접근 방식보다 우수하다.
* 즉, 생성 모델이 최상일 때 Generative replay를 통해 네트워크를 학습시키는 것은 전체 데이터셋을 동시에 학습시키는 것과 같다.
* 기본 실험으로, 섹션 4.1 에서는 망각의 정도를 조사하기위해 독립 task에 대한 네트워크를 순차적으로 훈련한다.
* 섹션 4.2 에서는 서로 다른 두개의 관련 도메인에서 네트워크를 학습시킴. 이는 generative replay가 알려진 다른 구조와도 호환 가능하다는 것을 보여준다.
* 섹션 4.3 에서는 학습데이터의 분리된 하위 집합에 대한 네트워크를 학습시킴으로써, 우리의 scholar 네트워크가 meta task를 수행하기 위해 다양한 task로 부터 지식을 수집할 수 있음을 보여준다.
