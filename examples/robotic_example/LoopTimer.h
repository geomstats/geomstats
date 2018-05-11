//LoopTimer.h

#ifndef SAI_LOOPTIMER_H_
#define SAI_LOOPTIMER_H_

#include <string>
#include <iostream>
#include <signal.h>

#define USE_CHRONO

#ifdef USE_CHRONO
#include <chrono>
#include <thread>
#else // USE_CHRONO

#include <unistd.h>
#include <time.h>
#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

#endif // USE_CHRONO

/** \brief Accurately time a loop to set frequency.
 *
 */
class LoopTimer {

public:

	LoopTimer() {}

	virtual ~LoopTimer() {}

	/** \brief Set the loop frequency
	 * \param frequency The loop frequency that will be used for LoopTimer::run()
	 */
	void setLoopFrequency (double frequency);

	/** \brief Initialize the timing loop, if using your own while loop. call before waitForLoop.
	 * \param initial_wait_nanoseconds The delay before waitForNextLoop will return the first time
	 */
	void initializeTimer(unsigned int initial_wait_nanoseconds = 0);

	/** \brief Wait for next loop. Use in your while loop. Not needed if using LoopTimer::run().
	 * \return true if a wait was required, and false if no wait was required. */
	bool waitForNextLoop();

	/** \brief Number of loops since calling run. */
	unsigned long long elapsedCycles();

	/** \brief Time when waitForNextLoop was last called */
	double loopTime();

	/** \brief Elapsed computer time since calling initializeTimer() or run() in seconds. */
	double elapsedTime();

	/** \brief Elapsed simulation time since calling initializeTimer() or run() in seconds. */
	double elapsedSimTime();

#ifndef USE_CHRONO
	/** \brief Time when waitForNextLoop was last called */
	void loopTime(timespec& t);

	/** \brief Elapsed time since calling initializeTimer() or run() in seconds. */
	void elapsedTime(timespec& t);
#endif  // USE_CHRONO

	/** \brief Run a loop that calls the user_callback(). Blocking function.
	 * \param userCallback A function to call every loop.
	 */
	void run(void (*userCallback)(void));

	/** \brief Stop the loop, started by run(). Use within callback, or from a seperate thread. */
	void stop();

	/** \brief Add a ctr-c exit callback.
	 * \param userCallback A function to call when the user presses ctrl-c.
	 */
	static void setCtrlCHandler(void (*userCallback)(int)) {
		struct sigaction sigIntHandler;
		sigIntHandler.sa_handler = userCallback;
		sigemptyset(&sigIntHandler.sa_mask);
		sigIntHandler.sa_flags = 0;
		sigaction(SIGINT, &sigIntHandler, NULL);
	}


	/** \brief Set the thread to a priority of -19. Priority range is -20 (highest) to 19 (lowest) */
	// static void setThreadHighPriority();

	/** \brief Set the thread to real time (FIFO). Thread cannot be preempted.
	 *  Set priority as 49 (kernel and interrupts are 50).
	 * \param MAX_SAFE_STACK maximum stack size in bytes which is guaranteed safe to access without faulting
	 */
	// static void setThreadRealTime(const int MAX_SAFE_STACK = 8*1024);

protected:

#ifndef USE_CHRONO
	inline void getCurrentTime(timespec &t_ret);

	inline void nanoSleepUntil(const timespec &t_next, const timespec &t_now);
#endif  // !USE_CHRONO

	static void printWarning(const std::string& message) {
		std::cout << "WARNING. LoopTimer. " << message << std::endl;
	}

	volatile bool running_ = false;

#ifdef USE_CHRONO
	std::chrono::high_resolution_clock::time_point t_next_;
	std::chrono::high_resolution_clock::time_point t_curr_;
	std::chrono::high_resolution_clock::time_point t_start_;
	std::chrono::high_resolution_clock::duration t_loop_;
	std::chrono::nanoseconds ns_update_interval_;
#else  // USE_CHRONO
	struct timespec t_next_;
	struct timespec t_curr_;
	struct timespec t_start_;
	struct timespec t_loop_;
	unsigned int ns_update_interval_ = 1e9 / 1000; // 1000 Hz
#endif  // USE_CHRONO

	unsigned long long update_counter_ = 0;

};

#endif /* SAI_LOOPTIMER_H_ */
