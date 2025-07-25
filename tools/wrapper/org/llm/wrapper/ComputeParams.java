package org.llm.wrapper;

public class ComputeParams {
    // Thread configuration

    /** N number of threads to use during generation (default: -1) */
    public int threads = -1;

    /** N number of threads to use during batch and prompt processing */
    public int threadsBatch = -1;

    /** M CPU affinity mask: arbitrarily long hex */
    public String cpuMask = "";

    /** lo-hi range of CPUs for affinity */
    public String cpuRange;

    /** <0|1> use strict CPU placement */
    public boolean cpuStrict = false;

    /** N set process/thread priority: low(-1), normal(0), medium(1), high(2), realtime(3) */
    public int priority = 0;

    /** <0...100> use polling level to wait for work (0 - no polling) */
    public int poll = 50;

    // Batch processing thread configuration

    /** M CPU affinity mask for batch processing */
    public String cpuMaskBatch;

    /** lo-hi ranges of CPUs for batch affinity */
    public String cpuRangeBatch;

    /** <0|1> use strict CPU placement for batch */
    public boolean cpuStrictBatch;

    /** N set batch process/thread priority */
    public int priorityBatch = 0;

    /** <0|1> use polling to wait for work in batch */
    public boolean pollBatch;

    // Constructor to handle defaults
    public ComputeParams() {
        // Set defaults that depend on other fields
        this.cpuMaskBatch = this.cpuMask;
        this.cpuStrictBatch = this.cpuStrict;
        this.pollBatch = this.poll > 0;
    }

    /**
     * Helper method to apply thread defaults if not explicitly set
     * @param defaultThreads The default number of threads to use
     */
    public void applyThreadDefaults(int defaultThreads) {
        if (threads == -1) {
            threads = defaultThreads;
        }
        if (threadsBatch == -1) {
            threadsBatch = threads;
        }
    }
}
