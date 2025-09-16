
        logger.info(f"Completed process_client_sow_from_end_to_end for client {client_id}")
        print("narrative is here>>>", existing_result)
        
        # Debug: Print the structure of existing_result
        if isinstance(existing_result, dict):
            print(f"DEBUG: Top-level keys in existing_result: {list(existing_result.keys())}")
            if 'narrative_output' in existing_result:
                narrative_output = existing_result['narrative_output']
                print(f"DEBUG: narrative_output keys: {list(narrative_output.keys()) if isinstance(narrative_output, dict) else 'Not a dict'}")
                if isinstance(narrative_output, dict) and 'narrative' in narrative_output:
                    narrative = narrative_output['narrative']
                    print(f"DEBUG: narrative type: {type(narrative)}, length: {len(narrative) if narrative else 0}")
                    print(f"DEBUG: narrative content preview: {str(narrative)[:200] if narrative else 'None'}")
        
