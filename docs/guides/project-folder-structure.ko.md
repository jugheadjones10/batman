# 프로젝트 폴더 구조

Batman은 저장소 루트를 기준으로 `data/projects/` 아래에 프로젝트 데이터를 저장합니다.
각 프로젝트에는 자체 동영상, 프레임, 라벨, 학습 실행 결과, 추론 결과가 있습니다.

## 프로젝트 내부

선택적으로 생성되는 산출물을 포함한 대표적인 구조입니다. 꺾쇠괄호 안의 이름은
실제 이름을 대신하는 자리표시자입니다. 폴더나 파일은 해당 작업 흐름이 실행된 후에만
생성될 수 있으며, 가져온 데이터셋과 라벨 스냅샷이 모든 프로젝트에 있는 것은 아닙니다.

```text
data/projects/<project_name>/
├── project.json
├── videos/
│   ├── videos.json
│   ├── <uploaded_video>.mp4                 (또는 .mkv 등)
│   ├── <video_id>_proxy.mp4
│   └── thumbnails/
├── frames/
│   ├── video_1/
│   │   ├── frames.json
│   │   └── video_1_000000.jpg               (및 추가 프레임)
│   └── <import_source_key>/
│       ├── frames.json
│       └── <frame_id>.jpg
├── imports/
│   └── imports.json
├── labels/
│   ├── current/
│   │   ├── annotations.json
│   │   └── tracks.json
│   └── iteration_<n>/
├── exports/
│   └── coco/
│       ├── train/
│       │   ├── _annotations.coco.json
│       │   └── <image>.jpg
│       ├── valid/
│       │   ├── _annotations.coco.json
│       │   └── <image>.jpg
│       └── test/
│           ├── _annotations.coco.json
│           └── <image>.jpg
├── runs/
│   └── <run_name>/
│       ├── meta.json
│       ├── training_config.json
│       ├── class_info.json
│       ├── results.json
│       ├── checkpoint_best_total.pth
│       ├── last.ckpt
│       ├── training.log
│       ├── metrics.csv
│       ├── hparams.yaml
│       └── events.out.tfevents.<...>
└── inference/
    └── <run_name>/
        └── <video_id>/
            └── <inference_id>/
                ├── result.json
                └── detected.mp4
```



## 폴더 설명: 각 폴더에 들어가는 내용

아래의 모든 경로는 `data/projects/<project_name>/`을 기준으로 합니다.

| 폴더 또는 파일 | 내용과 용도 |
| --- | --- |
| `project.json` | 프로젝트 이름, 설명, 클래스, 설정, 개수 정보 및 기타 프로젝트 메타데이터입니다. |
| `videos/` | 업로드한 원본 동영상과 생성된 재생용 프록시 동영상입니다. `videos.json`은 동영상 ID를 해당 파일 및 메타데이터와 연결합니다. |
| `videos/thumbnails/` | 동영상을 탐색할 때 사용하는, 자동 생성된 미리보기 이미지입니다. |
| `frames/` | 동영상에서 추출하거나 데이터셋 가져오기를 통해 추가한 정지 이미지입니다. 라벨링과 데이터셋 준비에 사용하는 이미지가 들어갑니다. |
| `frames/<video_id>/` | 업로드한 동영상 하나에서 추출한 프레임입니다. `frames.json`에는 프레임 메타데이터가 기록되며, JPEG 파일에는 실제 이미지가 저장됩니다. |
| `frames/<import_source_key>/` | `roboflow_crane-hook_1` 또는 `coco_zoo_person_1`과 같이 가져온 소스 하나의 이미지와 `frames.json`이 들어갑니다. 기존 소스는 `1` 또는 `-1`과 같은 숫자 폴더 이름을 사용할 수 있습니다. |
| `imports/` | 가져온 데이터의 출처 정보입니다. `imports.json`에는 소스의 세부 정보와 식별자가 기록되며, 가져온 이미지 자체는 `frames/`에 저장됩니다. |
| `labels/` | 편집 가능한 주석과 저장된 라벨링 반복 작업이 들어갑니다. |
| `labels/current/` | `annotations.json`에 현재 사용 중인 라벨이 저장되며, `tracks.json`이 있으면 추적 메타데이터도 포함됩니다. 수동 수정과 자동 라벨링은 현재 사용 중인 주석을 업데이트합니다. |
| `labels/iteration_<n>/` | 반복 작업 흐름을 통해 생성된, 라벨링 반복 작업의 저장된 스냅샷입니다. |
| `exports/` | 프로젝트 이미지와 주석으로 준비한 데이터셋입니다. 기본 COCO 내보내기 경로는 `exports/coco/`입니다. |
| `exports/coco/train/` | 학습 이미지와 해당 COCO 주석으로, 모델이 학습하는 예제입니다. |
| `exports/coco/valid/` | 학습 중 진행 상황을 평가하는 데 사용하는 검증 이미지와 주석입니다. 디렉터리 이름은 `val`이 아니라 `valid`입니다. |
| `exports/coco/test/` | 평가를 위해 따로 분리해 둔 테스트 이미지와 주석입니다. 각 분할에 포함되는 내용은 내보내기 설정에 따라 달라집니다. |
| `runs/` | 각 실행별로 별도의 폴더에 모아 둔 학습 출력물입니다. |
| `runs/<run_name>/` | 모델 체크포인트, 학습 설정, 클래스 정보, 지표, 로그가 들어갑니다. `meta.json`은 백엔드 작업 상태를, `training_config.json`은 실행 명령 정보를, `class_info.json`은 모델 클래스를 기록합니다. `results.json`에는 평가 결과가 생성된 경우 해당 결과가 포함됩니다. 체크포인트와 로그 이름은 학습 백엔드 및 버전에 따라 달라지며, 트리에는 대표적인 예시가 나와 있습니다. |
| `inference/` | 학습된 모델이 프로젝트 동영상에 대해 수행한 예측을 저장합니다. |
| `inference/<run_name>/` | 특정 학습 실행과 관련된 결과입니다. 이 이름은 해당 예측을 `runs/<run_name>/`과 연결합니다. |
| `inference/<run_name>/<video_id>/` | 동영상 하나에 대한 결과로, 여러 추론 세션이 포함될 수 있습니다. |
| `inference/<run_name>/<video_id>/<inference_id>/` | `20260901_144813`과 같이 타임스탬프가 붙은 하나의 추론 세션입니다. `result.json`에는 설정, 통계, 프레임별 예측이 저장됩니다. 선택적으로 생성되는 `detected.mp4`는 동영상 위에 예측을 겹쳐 보여줍니다. 저장된 거리 보정 정보는 `result.json` 내부에 `z_calibration`으로 저장됩니다. |




### 사용 시 참고 사항

- 일반적인 흐름은 **videos/imports → frames → labels → exports → 학습 실행 → inference**입니다.
- 기존 프로젝트에는 작업 흐름에 해당하는 폴더가 비어 있거나 없을 수 있습니다. 예를 들어,
  `Phase 2 Z height`에는 현재 `inference/` 폴더와 `labels/current/`가 없습니다.
- 이전 추론 결과는 추론 세션 폴더 없이
  `inference/<run_name>/<video_id>/result.json`에 바로 저장되어 있을 수 있습니다.
- 메타데이터는 그 메타데이터가 설명하는 미디어와 함께 보관하세요. 이미지나 동영상
  파일만 추가해도 애플리케이션에 등록되는 것은 아니므로, 업로드 또는 가져오기 작업 흐름을 사용하세요.
- 이 가이드는 프로젝트 내부의 저장 구조를 설명합니다. `datasets/`와 `runs/` 같은
  저장소 수준의 폴더는 일부 독립 실행 작업 흐름에서 사용하거나 출력 경로를 명시적으로
  재지정할 때 사용하는 별도의 위치입니다.

작업 흐름에 관한 자세한 내용은 [데이터셋 가져오기](../cli/importer.md),
[학습](training.md), [추론](inference.md)을 참고하세요.
