package com.google.ai.edge.litert;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

public final class CompiledModel {
    public void close() {}

    private static final int DEFAULT_BUFFER_SIZE = 1024 * 1024;

    private CompiledModel() {}

    public static CompiledModel create(LiteRtEnvironment env, String modelPath, LiteRtOptions opts) {
        return new CompiledModel();
    }

    public InputBuffers createInputBuffers(int signatureIndex) {
        return new InputBuffers();
    }

    public OutputBuffers createOutputBuffers(int signatureIndex) {
        return new OutputBuffers();
    }

    public void run(InputBuffers inputBuffers, OutputBuffers outputBuffers) {
        int copyCount = Math.min(inputBuffers.size(), outputBuffers.size());
        for (int i = 0; i < copyCount; i++) {
            ByteBuffer in = inputBuffers.get(i);
            ByteBuffer out = outputBuffers.get(i);
            in.rewind();
            out.clear();
            int copyLen = Math.min(in.remaining(), out.remaining());
            byte[] data = new byte[copyLen];
            in.get(data);
            out.put(data);
            out.rewind();
        }
    }

    public static final class InputBuffers {
        private final List<ByteBuffer> buffers = new ArrayList<>();

        public ByteBuffer get(int index) {
            ensure(index);
            return buffers.get(index);
        }

        private void ensure(int index) {
            while (buffers.size() <= index) {
                buffers.add(ByteBuffer.allocateDirect(DEFAULT_BUFFER_SIZE).order(ByteOrder.nativeOrder()));
            }
        }

        int size() {
            return buffers.size();
        }
    }

    public static final class OutputBuffers {
        private final List<ByteBuffer> buffers = new ArrayList<>();

        public ByteBuffer get(int index) {
            ensure(index);
            return buffers.get(index);
        }

        private void ensure(int index) {
            while (buffers.size() <= index) {
                buffers.add(ByteBuffer.allocateDirect(DEFAULT_BUFFER_SIZE).order(ByteOrder.nativeOrder()));
            }
        }

        int size() {
            return buffers.size();
        }
    }
}
