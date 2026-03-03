package com.google.ai.edge.litert;

public final class LiteRtOptions {
    private final Accelerator accelerator;

    private LiteRtOptions(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    public Accelerator getAccelerator() {
        return accelerator;
    }

    public static final class Builder {
        private Accelerator accelerator = Accelerator.CPU;

        public Builder setAccelerator(Accelerator accelerator) {
            this.accelerator = accelerator;
            return this;
        }

        public LiteRtOptions build() {
            return new LiteRtOptions(accelerator);
        }
    }
}
