                Customer Question
                       |
                       v
                LangGraph Agent
                       |
             Classify customer intent
                       |
              +--------+--------+
              |                 |
              v                 v
       Need account data?   Need service request?
              |                 |
              v                 v
        account.py         service_request.py
              |                 |
              v                 v
       Account information  Request ID, queue,
                            and SLA
              |                 |
              +--------+--------+
                       |
                       v
                Agent processes
                  tool results
                       |
                       v
                AgentResponse
